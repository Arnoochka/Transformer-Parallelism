from torch.nn import Module
from torch import Tensor
import torch
from .ParallelModule import ParallelModule
from typing import Any, List, Literal 
from enum import Enum

class ParallelTypes(Enum):
    tensor = 'tp'
    pipeline = 'pp'
    moe = 'moe'
    sequence = 'sequence'



class ModuleEngine(Module):
    def __init__(self,
                 module: ParallelModule,
                 parallel_types: List[ParallelTypes]):
        super().__init__()
        self.module = module
        self.parallel_types = parallel_types
        
    def forward(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs)
    
    def generate(self, *args, **kwargs) -> Any:
        self.module.generate(*args, **kwargs)
        if ParallelTypes.pipeline in self.parallel_types:
            pass
            
    def reset_fake_generators(self) -> None:
        pass
            