import torch
import json
from torch import nn, Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Optional, Union, Callable
from transformers import PreTrainedModel, AutoTokenizer
from .BenchmarkStats import BenchmarkStats
from .Tracker import Tracker
from .GenerationFunc import GenerationFunc
from mytransformers.utils import Logger, get_model_size
import pandas as pd

def get_synchronize_func(group: ProcessGroup):
    def synchronize():
        dist.barrier(group)
    return synchronize

class BenchmarkModel:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: AutoTokenizer,
                 generate_func: GenerationFunc,
                 batch_func: Callable,
                 warm_up: bool = False,
                 model_name: Optional[str] = None,
                 description: Optional[str] = None,
                  max_prompt_len: int = 64,
                 max_new_tokens: int = 128,
                 dtype: torch.dtype = torch.float32,
                 save_model_config: bool = False,
                 save_stats: bool = False,
                 save_dir: str = ""):
        torch.set_default_dtype(dtype)
        
        if model_name is None:
            model_name = getattr(model.config, "_name_or_path", "unknown").split("/")[-1]
            
        self.model = model
        self.tokenizer = tokenizer
        self.generation_func = generate_func
        self.batch_func = batch_func
        self.save_stats = save_stats
        self.save_dir = save_dir
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self.warm_up = warm_up
        
        self.stats = {
            "description": description,
            "model_name": model_name,
            "model_size": get_model_size(self.model),
            "dtype": str(dtype).split(".")[-1],
            "max_prompt_len": max_prompt_len,
            "max_new_tokens": max_new_tokens,
            "dtype": str(dtype).split(".")[-1]
            }
        
        if save_model_config:
            config_path = save_dir + f"{model_name}_config.json"
            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump(model.config.to_dict(), file, ensure_ascii=False, indent=4)

            self.stats['model_config_dir'] = config_path
        else:
           self.stats['model_config_dir'] = None 
        
    def __call__(self,
                 prompts: List[str],
                 batch_size: int,
                 **generate_kwargs) -> BenchmarkStats:

        sync_func = get_synchronize_func(None)
        tracker = Tracker(sync_func=sync_func)
        self.stats['data_size'] = len(prompts)
        
        self.stats['batch_size'] = batch_size
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if self.warm_up:
            batches = self.batch_func(prompts, batch_size, self.max_prompt_len, self.tokenizer)
            self.generation_func(
                self.model,
                batches,
                self.max_new_tokens,
                **generate_kwargs)

        batches = self.batch_func(prompts, batch_size, self.max_prompt_len, self.tokenizer)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tracker.start()
        self.generation_func(
            self.model,
            batches,
            self.max_new_tokens,
            **generate_kwargs
        )

        inference_stats = tracker.stop()

        self.calculate_statistics(inference_stats)
        
        if self.save_stats:
            with open(self.save_dir + f"{self.stats['model_name']}_stats.json", 'w', encoding='utf-8') as file:
                json.dump(self.stats, file, ensure_ascii=False, indent=4)   
                
        return BenchmarkStats(**self.stats)
    
    def calculate_statistics(self, inference_stats: pd.DataFrame) -> None:
        total_time = inference_stats.loc["stop", "time"]

        total_tokens = self.stats["data_size"] * self.stats["max_new_tokens"]
        throughput = total_tokens / total_time
        max_memory_per_device = []
        for col in inference_stats.columns:
            if col.startswith("max_memory_gpu_"):
                max_memory_per_device.append(
                    inference_stats[col].max()
                )
                
        self.stats["inference_time"] = total_time
        self.stats["throughput"] = throughput
        self.stats["max_memory_allocated_per_device"] = max_memory_per_device
