from mytransformers.parallel.ParallelModule import ParallelModule
import torch
from torch.nn import Module
from enum import Enum
from typing import Any

class PipeRole(Enum):
    dummy = "dummy"
    recv = "recv"
    compute = "compute"
    computeAndSend = "computeAndSend"
    start = "start"
    end = "end"
    none = "none"

class PipeModule(ParallelModule):   
    """
    Базовый класс для модулей  конвейерного параллелизма
    """
    def __init__(self, role: PipeRole, module: Module):
        super().__init__()
        if role == PipeRole.none:
            raise AttributeError(f"role is none")
        self.role = role
        self.module = module
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs)
        
    def __repr__(self) -> str:
        repr = super().__repr__()
        name = self.__class__.__name__
        sub_module = ''.join(repr.split(f"{name}(")[1:])
        return f"{name}(role={self.role}{sub_module}"


        
    