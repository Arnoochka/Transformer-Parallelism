from typing import List, Optional, Union
from dataclasses import dataclass

@dataclass
class BenchmarkStats:
    description: Optional[str]
    tool: Optional[str]
    model_name: str
    model_size: float
    dtype: str
    parallelism: Optional[Union[str, List[str]]]
    model_config_dir: Optional[str]
    max_memory_allocated_per_device: List[float]
    max_memory_allocated_generator_per_device: List[float]
    max_memory_allocated_inference_per_device: List[List[float]]
    data_size: int
    batch_size: int
    max_prompt_len: int
    max_new_tokens_list: List[int]
    generate_time: float
    inference_time: List[float]
    throughput: List[float]
    
    def __str__(self) -> str:
        results = [f"Description: {self.description}"] \
            if self.description is not None else []
        results.append('-------Benchmark results-------') 
        if self.tool is not None:
            results.append(f"tool: {self.tool}")
        results.append(f"model name: {self.model_name}")
        results.append(f"model size: {self.model_size:.3f}, dtype: {self.dtype}")
        if self.parallelism is not None:
            if isinstance(self.parallelism, str):
               results.append(f"parallelism type: {self.parallelism}")
            else:
                results.append(f"parallelism types: {', '.join(self.parallelism)}") 
        if self.model_config_dir is not None:
            results.append(f"model config dir: {self.model_config_dir}")
        results.append(f"mean inference time (s): {sum(self.inference_time) / len(self.inference_time):.3f}")
        results.append(f"mean throughput (token/s): {sum(self.throughput) / len(self.throughput):.3f}")
        results.append(f"data size: {self.data_size}, batch size: {self.batch_size}, max len: {self.max_prompt_len}")
        results.append(f"max new tokens: {self.max_new_tokens_list}")
        
        indent = "\n    "  
        if len(self.max_new_tokens_list) == 1 and self.inference_time:
            results += self._get_loop_statistic(self.inference_time[0],
                                                self.throughput[0],
                                                indent=indent[1:])
        else:
            for idx, max_new_tokens in enumerate(self.max_new_tokens_list):
                results += self._get_loop_statistic(self.inference_time[idx],
                                                    self.throughput[idx],
                                                    max_new_tokens,
                                                    indent=indent[1:])   
                  
        results.append(f"\nmax memory allocated per device for generator (GB):\
            {self._get_statistic_from_list(self.max_memory_allocated_generator_per_device, indent=indent)}")
        results.append(f"\nmax memory allocated per device for inference (GB):\
            {self._get_statistic_from_list(self.max_memory_allocated_inference_per_device, indent=indent)}")
        results.append(f"\nmax memory allocated per device (GB):\
            {self._get_statistic_from_list(self.max_memory_allocated_per_device, indent=indent)}")
                
        return "\n".join(results)

            
    def _get_loop_statistic(self,
                           inference_time: float,
                           throughput: float,
                           max_new_tokens: Optional[int] = None,
                           indent: str = "") -> List[str]:
        results = [f"---max_new_tokens: {max_new_tokens}---"] if max_new_tokens is not None else []
        results.append(indent + f"inference time (s): {inference_time:.3f}")
        results.append(indent + f"throughput (token/s): {throughput:.3f}")
        
        return results
        
    
    def _get_statistic_from_list(self,
                                 memory_list: List[float],
                                 indent: str = "") -> str:
        if not memory_list:
            return "No data"
        memory_entries = [
            indent + f"device {idx}: {value:.3f}" 
            for idx, value in enumerate(memory_list)
        ]
        memory_info = "".join(memory_entries)
        return memory_info