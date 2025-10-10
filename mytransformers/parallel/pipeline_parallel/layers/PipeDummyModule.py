import torch
from torch import Tensor
from .PipeModule import PipeModule, PipeRole

class PipeDummyModule(PipeModule):
    def __init__(self, device: torch.device):
        super().__init__(PipeRole.dummy)
        self.device = device
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Tensor:
        for arg in args:
            if isinstance(arg, Tensor):
                return arg
        for kwarg in kwargs.values():
            if isinstance(kwarg, Tensor):
                return kwarg
        return torch.zeros(1, device=self.device)