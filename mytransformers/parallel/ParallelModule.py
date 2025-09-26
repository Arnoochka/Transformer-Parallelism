from torch.nn import Module
from torch import Tensor
import torch

class ParallelModule(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, *args, **kwargs) -> Tensor:
        return torch.empty(1)