import torch
import json
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Optional, Union
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer
from .BenchmarkStats import BenchmarkStats
from .Tracker import Tracker
from mytransformers.utils import logger, get_model_size

def get_synchronize_func(group: ProcessGroup):
    def synchronize():
        dist.barrier(group)
    return synchronize

class BenchmarkModel:
    def __init__(self,
                 model_name: str,
                 model_getter: PreTrainedModel,
                 generator: torch.nn.Module,
                 tokenizer: AutoTokenizer,
                 config: PretrainedConfig,
                 loops: int = 1,
                 batch_size: int = 1,
                 max_len: int = 256,
                 description: Optional[str] = None,
                 tool: Optional[str] = None,
                 parallelism: Optional[Union[str, List[str]]] = None,
                 dtype: torch.dtype = torch.float32,
                 save_model_config: bool = True):
        torch.set_default_dtype(dtype)
        self.model_name = model_name
        self.model = model_getter.from_pretrained(model_name,
                                                  config=config,
                                                  torch_dtype=dtype)
        self.generator = generator
        self.tokenizer = tokenizer.from_pretrained(model_name)
        self.loops = loops
        self.batch_size = batch_size
        self.max_len = max_len
        
        self.stats = {
            "description": description,
            "parallelism": parallelism,
            "tool": tool,
            "model_name": model_name,
            "model_size": get_model_size(self.model),
            "dtype": str(dtype).split(".")[-1],
            "model_config_dir": None,
            "loops": loops,
            "batch_size": batch_size,
            "max_len": max_len}
        
        if save_model_config:
            with open(f"{model_name}.json", 'w', encoding='utf-8') as file:
                json.dump(config.to_dict(), file, ensure_ascii=False, indent=4)   
                
            self.stats['model_config_dir'] = model_name  
        
    def __call__(self,
                 input_data: List[str],
                 group: ProcessGroup,
                 print_output_num: int = 0) -> BenchmarkStats:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        sync_func = get_synchronize_func(group)
        tracker = Tracker(group, sync_func)
        self.stats['data_size'] = len(input_data)
        logger("start benchmark", rank)
        tracker.start() 
        model = self.generator(self.model, group)
        tracker.snapshot("generator")
        
        inference_times = []
        throughputs = []
        memory_allocated_inference = []
        
        for loop in range(self.loops):
            tracker.snapshot(f"loop:{loop} start")
            for batch in self.batches(input_data):
                output = model.generate(batch, max_len=self.max_len)
            tracker.snapshot(f"loop:{loop} stop")
            
            stats = tracker.stop()
            inference_time = stats['time'][-1] - stats['time'][-2]
            inference_times.append(inference_time)
            throughput = (self.batch_size * self.max_len) / inference_time
            throughputs.append(throughput)
            memory_allocated_inference.append([mem.tolist() for mem in stats['memory'][-1]])
            
        logger("stop benchmark", rank)
        if print_output_num > 0:
            decoded_output = self.tokenizer.decode(output[0][:print_output_num])
            logger(f"output: {decoded_output}", rank)
            
        final_stats = tracker.stop()
        
        # Сбор памяти
        max_memory_allocated_per_device = [max(mem).item() for mem in final_stats['max_memory']]
        memory_allocated_generator_per_device = [mem[0].item() for mem in final_stats['memory'][1]]
        
        return BenchmarkStats(
            description=self.stats["description"],
            tool=self.stats["tool"],
            model_name=self.stats["model_name"],
            model_size=self.stats["model_size"],
            dtype=self.stats["dtype"],
            parallelism=self.stats["parallelism"],
            model_config_dir=self.stats["model_config_dir"],
            max_memory_allocated_per_device=max_memory_allocated_per_device,
            memory_allocated_generator_per_device=memory_allocated_generator_per_device,
            memory_allocated_inference_per_device=memory_allocated_inference,
            loops=self.stats["loops"],
            data_size=self.stats["data_size"],
            batch_size=self.stats["batch_size"],
            max_len=self.stats["max_len"],
            generate_time=sum(inference_times),
            inference_time=inference_times,
            throughput=throughputs
        )
        
    def batches(self, input_data: List[str]):
        for i in range(0, len(input_data), self.batch_size):
            batch_data = input_data[i:i + self.batch_size]
            encoded_batch = []
            for text in batch_data:
                encoded = self.tokenizer.encode(text, return_tensors='pt')
                if encoded.size(1) > self.max_len:
                    encoded = encoded[:, :self.max_len]
                else:
                    pad_len = self.max_len - encoded.size(1)
                    encoded = torch.nn.functional.pad(encoded, (0, pad_len), value=self.tokenizer.pad_token_id or 0)
                encoded_batch.append(encoded)
            
            batch_tensor = torch.cat(encoded_batch, dim=0)
            batch_tensor = batch_tensor.to(torch.cuda.current_device())
            yield batch_tensor
