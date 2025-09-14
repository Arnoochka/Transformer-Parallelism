from typing import List, Optional, Union
from dataclasses import dataclass

@dataclass
class BenchmarkStats:
    description: Optional[str]
    tool: Optional[str]
    model_name: Optional[str]
    model_size: float
    dtype: str
    parallelism: Optional[Union[str, List[str]]]
    model_config_dir: Optional[str]
    
    max_memory_allocated_per_device: List[float]
    memory_allocated_generator_per_device: List[float]
    memory_allocated_inference_per_device: List[float]
    
    loops: int
    batch_size: int
    generate_time: float
    inference_token_time: Union[List[float], List[List[float]]]
    inference_time: Union[List[float], List[List[float]]]
    throughput: Union[float, List[float]]
    
    max_print_items: int = 10
    
    def __str__(self) -> str:
        results = [f"Description: {self.description}"] \
            if self.description is not None else []
        results.append(['-------Benchmark results-------']) 
        if self.tool is not None:
            results.append(f"tool: {self.tool}")
        if self.model_name is not None:
            results.append(f"model name: {self.model_name}")
        results.append(f" model size: {self.model_size}, dtype: {self.dtype}")
        if self.parallelism is not None:
            if isinstance(self.parallelism, str):
               results.append(f"parallellism type: {self.parallelism}")
            else:
                results.append(f"parallellism types: {", ".join(self.parallelism)}") 
        if self.model_config_dir is not None:
            results.append(f"model config dir: {self.model_config_dir}")
        results.append(f"batch size: {self.batch_size}")
        results.append(f"max memory allocated per device (GB): \
            {self._get_statistic_from_list(self.max_memory_allocated_per_device)}")
        results.append(f"max memory allocated per device for generator (GB): \
            {self._get_statistic_from_list(self.memory_allocated_generator_per_device)}")
        results.append(f"max memory allocated per device for inference (GB): \
            {self._get_statistic_from_list(self.memory_allocated_inference_per_device)}" )
        results.append(f"generate time (s): {self.generate_time}")
        results.append(f"num loops: {self.loops}")
        results.append(f"batch size: {self.batch_size}")
        
        if self.loops == 1:
            results += self._get_loop_statistic(self.inference_token_time,
                                                self.inference_time,
                                                self.throughput)
        else:
            for loop in range(self.loops):
                results += self._get_loop_statistic(self.inference_token_time[loop],
                                                    self.inference_time[loop],
                                                    self.throughput[loop],
                                                    loop=loop)
                
        return "\n".join(results)

            
    def _get_loop_statistic(self,
                           inference_token_time: float,
                           inference_time: List[float],
                           throughput: float,
                           loop: Optional[int] = None) -> List[str]:
        results = [f"---loop: {loop}---"] if loop is not None else []
        results.append(f"inference token time (s): \
            {self._get_statistic_from_list(inference_token_time)}")
        results.append(f"inference time (s): {inference_time}")
        results.append(f"throughput (token/s): {throughput}")
        
        return results
        
        
    def _get_statistic_from_list(self,
                                 memory_list: List[float]) -> str:
        memory_entries = [
            f"{idx}: {value}" 
            for idx, value in enumerate(memory_list[:self.max_print_items])
        ]
        memory_info = " ".join(memory_entries)
        return memory_info
    