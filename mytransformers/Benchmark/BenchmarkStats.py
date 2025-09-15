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
    memory_allocated_generator_per_device: List[float]
    memory_allocated_inference_per_device: List[List[float]]
    loops: int
    data_size: int
    batch_size: int
    max_len: int
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
        results.append(f"model size: {self.model_size}, dtype: {self.dtype}")
        if self.parallelism is not None:
            if isinstance(self.parallelism, str):
               results.append(f"parallelism type: {self.parallelism}")
            else:
                results.append(f"parallelism types: {', '.join(self.parallelism)}") 
        if self.model_config_dir is not None:
            results.append(f"model config dir: {self.model_config_dir}")
        results.append(f"mean inference time (s): {sum(self.inference_time) / len(self.inference_time) if self.inference_time else 0}")
        results.append(f"mean throughput (token/s): {sum(self.throughput) / len(self.throughput) if self.throughput else 0}")
        results.append(f"num loops: {self.loops}, data size: {self.data_size}, batch size: {self.batch_size}, max len: {self.max_len}")
        
        if self.loops == 1 and self.inference_time:
            results += self._get_loop_statistic(self.inference_time[0] if isinstance(self.inference_time, list) and len(self.inference_time) > 0 else self.inference_time,
                                                self.throughput[0] if isinstance(self.throughput, list) and len(self.throughput) > 0 else self.throughput,
                                                self.memory_allocated_inference_per_device[0] if isinstance(self.memory_allocated_inference_per_device, list) and len(self.memory_allocated_inference_per_device) > 0 else [])
        else:
            for loop in range(min(self.loops, len(self.inference_time) if self.inference_time else 0)):
                results += self._get_loop_statistic(self.inference_time[loop],
                                                    self.throughput[loop],
                                                    self.memory_allocated_inference_per_device[loop] if loop < len(self.memory_allocated_inference_per_device) else [],
                                                    loop=loop)
                
        results.append(f"max memory allocated per device for generator (GB): {self._get_statistic_from_list(self.memory_allocated_generator_per_device)}")
        results.append(f"max memory allocated per device (GB): {self._get_statistic_from_list(self.max_memory_allocated_per_device)}")
                
        return "\n".join(results)

            
    def _get_loop_statistic(self,
                           inference_time: float,
                           throughput: float,
                           memory_allocated_inference_per_device: List[float],
                           loop: Optional[int] = None) -> List[str]:
        
        results = [f"---loop: {loop}---"] if loop is not None else []
        results.append(f"   inference time (s): {inference_time}")
        results.append(f"   throughput (token/s): {throughput}")
        results.append(f"max memory allocated per device for inference (GB): {self._get_statistic_from_list(memory_allocated_inference_per_device)}")
        
        return results
        
        
    def _get_statistic_from_list(self,
                                 memory_list: List[float]) -> str:
        if not memory_list:
            return "No data"
        memory_entries = [
            f"{idx}: {value:.4f}" 
            for idx, value in enumerate(memory_list[:self.max_print_items])
        ]
        memory_info = " ".join(memory_entries)
        return memory_info