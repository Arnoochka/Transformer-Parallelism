from .FakeModule import FakeModule
from typing import Optional, Tuple, List
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
        return self.get_tensor(self.tensor_shape)
    
    def get_tensor(self, tensor_shape: Tuple[int]) -> Tensor:
        return torch.zeros(tensor_shape, device=self.device, dtype=self.dtype)
    
    def set_gen_args(self, tensor_shape: Tuple[int]) -> None:
        self.tensor_shape = tensor_shape
        
        
class FakeTupleTensorModule(FakeTensorModule):
    """
    генерирует кортеж фейковых тензоров необходимой размерности. 
    
    Args:
        device (torch.device): устройство, на котором должен быть выходной тензор
        dtype (torch.dtype): тип данных, с которым должен быть выходной тензор (по умолчанию акой же, как и тип данных, выставленный по умолчанию в torch)
    """
    def __init__(self,
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(dtype)
        self.tensor_shapes = None
        self.device = device
    
    def forward(self, *args, **kwargs) -> Tuple[Tensor]:
        outputs = [self.get_tensor(tensor_shape)
                   for tensor_shape in self.tensor_shapes]
        return tuple(outputs)
    
    def set_gen_args(self, tensor_shapes: List[Tuple[int]]) -> None:
        self.tensor_shapes = tensor_shapes
    
