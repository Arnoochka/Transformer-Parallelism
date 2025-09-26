import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional

class PipeDummyModule(Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
    def forward(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, Tensor):
                return arg
        for kwarg in kwargs.values():
            if isinstance(kwarg, Tensor):
                return kwarg
        return torch.zeros(1, device=self.device)