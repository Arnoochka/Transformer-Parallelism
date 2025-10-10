from .PipeModule import PipeModule, PipeRole
from torch.nn import Module
from torch import Tensor
import torch

class PipeComputeModule(PipeModule):
    def __init__(self,
                 role: PipeRole,
                 module: Module):
        super().__init__(role)
        self.module = module 
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Tensor:
        return self.module(*args, **kwargs)


        
    