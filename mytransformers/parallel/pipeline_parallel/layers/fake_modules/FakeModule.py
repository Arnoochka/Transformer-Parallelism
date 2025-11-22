from typing import Any, Optional
import torch
from torch.nn import Module

class FakeModule(Module):
    """
    базовый класс для "фейкового" слоя
    
    Args:
        device (torch.device): устройство, на котором должен быть выходной тензор
        dtype (torch.dtype): тип данных, с которым должен быть выходной тензор (по умолчанию акой же, как и тип данных, выставленный по умолчанию в torch)
    """
    def __init__(self,
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> Any:
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"
    
    def set_gen_args(self, *args) -> None:
        return None