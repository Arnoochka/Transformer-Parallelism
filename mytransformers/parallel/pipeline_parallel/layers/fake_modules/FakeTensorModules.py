from .FakeModule import FakeModule
from typing import Optional, Tuple
import torch
from torch import Tensor

class FakeTensorModule(FakeModule):
    """
    генерирует фейковый тензор необходимой размерности
    
    Args:
        device (torch.device): устройство, на котором должен быть выходной тензор
        dtype (torch.dtype): тип данных, с которым должен быть выходной тензор (по умолчанию акой же, как и тип данных, выставленный по умолчанию в torch)
    """
    def __init__(self,
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(device, dtype)
        self.tensor_shape = None
        
    def forward(self, *args, **kwargs) -> Tensor:
        return torch.zeros(self.tensor_shape, device=self.device, dtype=self.dtype)
    
    def set_gen_args(self, tensor_shape: Tuple[int]) -> None:
        self.tensor_shape = tensor_shape
    
