from typing import Optional, List, Tuple
import torch
from torch import Tensor
from .FakeModule import FakeModule

class FakeTupleTensorModule(FakeModule):
    """
    генерирует кортеж фейковых тензоров необходимой размерности. 
    
    Args:
        tensor_shape (List[Tuple[int]]): начальные размеры выходных тензоров
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
    
    def get_tensor(self, tensor_shape: Tuple[int]) -> Tensor:
        return torch.zeros(tensor_shape, device=self.device, dtype=self.dtype)
    
    def set_gen_args(self, tensor_shapes: List[Tuple[int]]) -> None:
        self.tensor_shapes = tensor_shapes
    
    
class FakeTupleOptionalTensorModule(FakeTupleTensorModule):
    """
    генерирует кортеж фейковых тензоров необходимой размерности. Однако, некоторые значения в кортеже могут быть None
    
    Args:
        tensor_shape (List[Optional[Tuple[int]]]): начальные размеры выходных тензоров
        device (torch.device): устройство, на котором должен быть выходной тензор
        dtype (torch.dtype): тип данных, с которым должен быть выходной тензор (по умолчанию акой же, как и тип данных, выставленный по умолчанию в torch)
    """
    def forward(self, *args, **kwargs) -> Tuple[Optional[Tensor]]:
        outputs = [self.get_tensor(tensor_shape) if tensor_shape is not None else None
                   for tensor_shape in self.tensor_shapes]
        return tuple(outputs)
        