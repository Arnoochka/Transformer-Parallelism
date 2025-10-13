from typing import Any, Optional
import torch
from torch.nn import ReLU

class FakeModule(torch.nn.Module):
    def __init__(self,
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> Any:
        return torch.zeros((1, 1, 1), device=self.device, dtype=self.dtype)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"