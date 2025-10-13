from .PipeModule import PipeModule, PipeRole
from torch.nn import Module
from typing import Any
import torch

class PipeComputeModule(PipeModule):
    def __init__(self,
                 role: PipeRole,
                 module: Module):
        super().__init__(role)
        self.module = module 
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs)


        
    