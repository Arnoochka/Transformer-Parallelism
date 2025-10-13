from torch.nn import Module
from torch import Tensor
import torch
from .ParallelModule import ParallelModule
from typing import Any, List, Callable
from enum import Enum

class ParallelTypes(Enum):
    tensor = 'tp_parallel'
    pipeline = 'pp_parallel'
    moe = 'moe_parallel'
    sequence = 'sequence_parallel'



class ParallelModuleEngine(Module):
    def __init__(self,
                 module: ParallelModule,
                 parallel_types: List[ParallelTypes],
                 reset_strategy: Callable):
        super().__init__()
        self.module = module
        self.parallel_types = parallel_types
        self.reset_strategy = reset_strategy
        
    def forward(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs)
    
    def generate(self, *args, **kwargs) -> Any:
        self.module.generate(*args, **kwargs)
        if ParallelTypes.pipeline in self.parallel_types:
            self.reset_strategy(self.module)
            
    def reset_fake_generators(self) -> None:
        pass
            