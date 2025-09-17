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
    loops: int
    data_size: int
    batch_size: int
    max_prompt_len: int
    max_new_tokens: int
    generate_time: float
    inference_time: List[float]
    throughput: List[float]
    
    max_print_items: int = 10
    
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
        results.append(f"num loops: {self.loops}, data size: {self.data_size}, batch size: {self.batch_size}, max len: {self.max_prompt_len}, max new tokens: {self.max_new_tokens}")
        
        indent = "\n    "  
        if self.loops == 1 and self.inference_time:
            results += self._get_loop_statistic(self.inference_time[0],
                                                self.throughput[0],
                                                indent=indent[1:])
        else:
            for loop in range(self.loops):
                results += self._get_loop_statistic(self.inference_time[loop],
                                                    self.throughput[loop],
                                                    loop=loop+1,
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
                           loop: Optional[int] = None,
                           indent: str = "") -> List[str]:
        results = [f"---loop: {loop}---"] if loop is not None else []
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
            for idx, value in enumerate(memory_list[:self.max_print_items])
        ]
        memory_info = "".join(memory_entries)
        return memory_info