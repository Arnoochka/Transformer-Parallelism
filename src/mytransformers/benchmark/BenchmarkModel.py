import torch
import json
from torch import nn, Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Optional, Union
from transformers import PreTrainedModel, AutoTokenizer
from .BenchmarkStats import BenchmarkStats
from .Tracker import Tracker
from mytransformers.parallel import ParallelModuleGenerator
from mytransformers.utils import Logger, get_model_size
import pandas as pd

def get_synchronize_func(group: ProcessGroup):
    def synchronize():
        dist.barrier(group)
    return synchronize

class BenchmarkModel:
    def __init__(self,
                 model: PreTrainedModel,
                 generator: ParallelModuleGenerator,
                 tokenizer: AutoTokenizer,
                 batch_size: Optional[int] = None,
                 max_prompt_len: int = 64,
                 max_new_tokens_list: List[int] = [128],
                 model_name: Optional[str] = None,
                 description: Optional[str] = None,
                 tool: Optional[str] = None,
                 parallelism: Optional[Union[str, List[str]]] = None,
                 dtype: torch.dtype = torch.float32,
                 save_model_config: bool = False,
                 save_stats: bool = False,
                 save_dir: str = ""):
        torch.set_default_dtype(dtype)
        if model_name is None:
            model_name = getattr(model.config, "_name_or_path", "unknown").split("/")[-1]
        self.model = model
        self.generator = generator
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens_list = max_new_tokens_list
        self.save_stats = save_stats
        self.save_dir = save_dir
        
        self.stats = {
            "description": description,
            "parallelism": parallelism,
            "tool": tool,
            "model_name": model_name,
            "model_size": get_model_size(self.model),
            "dtype": str(dtype).split(".")[-1],
            "max_prompt_len": max_prompt_len,
            "max_new_tokens_list": max_new_tokens_list}
        
        if save_model_config:
            with open(save_dir + f"{model_name}_config.json", 'w', encoding='utf-8') as file:
                json.dump(model.config.to_dict(), file, ensure_ascii=False, indent=4)   
                
            self.stats['model_config_dir'] = save_dir + f"{model_name}_config.json"
        else:
           self.stats['model_config_dir'] = None 
        
    def __call__(self,
                 prompts: List[str],
                 group: ProcessGroup,
                 print_output_num: int = 0) -> BenchmarkStats:
        world_size = dist.get_world_size(group)
        sync_func = get_synchronize_func(group)
        tracker = Tracker(group, sync_func)
        self.stats['data_size'] = len(prompts)
        Logger.log_main_device("start benchmark")

        tracker.start() 
        model = self.generator(self.model, torch.cuda.current_device())
        tracker.snapshot("generator")
        Logger.log_main_device(f"model:\n{[child for child in model.children()]}")
        # self.generate(model, prompts, self.max_new_tokens_list[-1])
        for max_new_tokens in self.max_new_tokens_list:
            tracker.snapshot(f"max new tokens:{max_new_tokens} start")
            output = self.generate(model, prompts, max_new_tokens)
            tracker.snapshot(f"max new tokens:{max_new_tokens} stop")
        Logger.log_main_device("stop benchmark")
        if print_output_num > 0:
            outputs = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            decoded_output = outputs[:print_output_num]
            Logger.log_main_device(f"output: {decoded_output}")

        final_stats = tracker.stop()
        Logger.log_main_device(final_stats)
        self.calculate_statistics(final_stats, world_size)
        
        if self.save_stats:
            with open(self.save_dir + f"{self.stats['model_name']}_stats.json", 'w', encoding='utf-8') as file:
                json.dump(self.stats, file, ensure_ascii=False, indent=4)   
                
        return BenchmarkStats(**self.stats)


    def batches(self, prompts: List[str]):
        if self.batch_size is None:
            self.batch_size = len(prompts)
            
        self.stats['batch_size'] = self.batch_size
            
        def _batch_encode(data):
            kwargs = {"return_tensors": "pt",
                      "padding": "max_length",
                      "max_length": self.max_prompt_len}
            input_tokens = self.tokenizer.batch_encode_plus(data, **kwargs)
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
            return input_tokens
        
        for i in range(0, len(prompts), self.batch_size):
            yield _batch_encode(prompts[i:i+self.batch_size])
            
    def generate(self, model: nn.Module,
                 prompts: List,
                 max_new_tokens: int) -> Tensor:
        for batch in self.batches(prompts):
            output = model.generate(**batch, max_new_tokens=max_new_tokens, use_cache=False)
        return output
    
    def calculate_statistics(self, final_stats: pd.DataFrame, world_size: int) -> None:
        inference_times = []
        throughputs = []
        max_memory_allocated_generator_per_device = []
        max_memory_allocated_inference_per_device = [0.0] * world_size
        for max_new_tokens in self.max_new_tokens_list:
            t_start = final_stats.loc[f"max new tokens:{max_new_tokens} start"]['time']
            t_stop  = final_stats.loc[f"max new tokens:{max_new_tokens} stop"]['time']
            inference_time = t_stop - t_start
            inference_times.append(inference_time)
            throughput = (self.batch_size * max_new_tokens) / inference_time
            throughputs.append(throughput)
        max_memory_allocated_per_device = [0.0] * world_size  
        max_memory_allocated_generator_per_device = [0.0] * world_size  
        for idx in range(world_size):
            max_memory_allocated_per_device[idx] = final_stats[f"max_memory_gpu_{idx}"].max()
            max_memory_allocated_generator_per_device[idx] = final_stats.loc['generator'][f"max_memory_gpu_{idx}"]
            max_memory_allocated_inference_per_device[idx] =\
                final_stats.loc[[f"max new tokens:{max_new_tokens} stop" 
                                 for max_new_tokens in self.max_new_tokens_list]][f"max_memory_gpu_{idx}"].max()
        self.stats["max_memory_allocated_per_device"] = max_memory_allocated_per_device
        self.stats["max_memory_allocated_generator_per_device"] = max_memory_allocated_generator_per_device
        self.stats["max_memory_allocated_inference_per_device"] = max_memory_allocated_inference_per_device
        self.stats["generate_time"] = final_stats.loc['generator']['time']
        self.stats["inference_time"] = inference_times
        self.stats["throughput"] = throughputs
