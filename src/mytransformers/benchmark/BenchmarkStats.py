from typing import List, Optional, Union
from dataclasses import dataclass

@dataclass
class BenchmarkStats:
    description: Optional[str]
    model_name: str
    model_size: float
    dtype: str
    model_config_dir: Optional[str]
    max_memory_allocated_per_device: List[float]
    data_size: int
    batch_size: int
    max_prompt_len: int
    max_new_tokens: int
    inference_time: float
    throughput: float
    
    def __str__(self) -> str:
        results = [f"Description: {self.description}"] \
            if self.description is not None else []
        results.append('-------Benchmark results-------') 
        results.append(f"model name: {self.model_name}")
        results.append(f"model size: {self.model_size:.3f}, dtype: {self.dtype}")
        if self.model_config_dir is not None:
            results.append(f"model config dir: {self.model_config_dir}")
        results.append(f"inference time (s): {self.inference_time:.3f}")
        results.append(f"throughput (token/s): {self.throughput:.3f}")
        results.append(f"data size: {self.data_size}, batch size: {self.batch_size}, max len: {self.max_prompt_len}, max new tokens: {self.max_new_tokens}")
        
        indent = "\n    "  
        results += self._get_loop_statistic(self.inference_time,
                                            self.throughput,
                                            indent=indent[1:])  
        results.append(f"\nmax memory allocated per device (GB):\
            {self._get_statistic_from_list(self.max_memory_allocated_per_device, indent=indent)}")
                
        return "\n".join(results)

            
    def _get_loop_statistic(self,
                           inference_time: float,
                           throughput: float,
                           indent: str = "") -> List[str]:
        results = []
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