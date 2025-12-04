from .PipeModule import PipeModule, PipeRole
from torch.nn import Module
from typing import Any
import torch

class PipeComputeModule(PipeModule):
    """
    Вычислительный модуль для конвейерного параллелизма.
    
    Задача: обработка слоем (module) входных данных
    
    Args:
        role (PipeRole): роль модуля (compute или recvAndCompute) 
        module (Module): слой, который должен произвести выччисление
    """
    def __init__(self, module: Module):
        super().__init__(PipeRole.compute, module)
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs)


        
    