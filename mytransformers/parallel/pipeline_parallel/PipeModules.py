from mytransformers.parallel.ParallelModule import ParallelModule
from torch.nn import ModuleList
from typing import Callable
from torch import Tensor
from enum import Enum

class PipeRole(Enum):
    dummy = 0
    comm = 1
    compute = 2
    computeAndcomm = 3

class PipeModule(ParallelModule):
    def __init__(self,
                 role: PipeRole,
                 func: Callable):
        super().__init__()
        self.role = role
        self.func = func
        
    def forward(self, x: Tensor) -> Tensor:
        return self.func(x)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(role={self.role.name})"


        
    