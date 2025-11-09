from .FakeModule import FakeModule
from typing import Optional, Tuple
import torch
from torch import Tensor

class FakeTensorModule(FakeModule):
    """
    генерирует фейковый тензор необходимой размерности
    
    Args:
        tensor_shape (Tuple[int]): размер выходного тензора
        device (torch.device): устройство, на котором должен быть выходной тензор
        dtype (torch.dtype): тип данных, с которым должен быть выходной тензор (по умолчанию акой же, как и тип данных, выставленный по умолчанию в torch)
    """
    def __init__(self,
                 tensor_shape: Tuple[int],
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(device, dtype)
        self.tensor_shape = tensor_shape
        
    def forward(self, *args, **kwargs) -> Tensor:
        return torch.zeros(self.tensor_shape, device=self.device, dtype=self.dtype)
        
        
class FakeSeqModule(FakeTensorModule):
    """
    генерирует фейковый тензор необходимой размерность. Однако, при каждой новой генерации увеличивает нужную размерность на 1
    
    Args:
        init_tensor_shape (Tuple[int]): начальный размер выходного тензора
        seq_dim (int): размерность, которая будет увеличиваться
        device (torch.device): устройство, на котором должен быть выходной тензор
        dtype (torch.dtype): тип данных, с которым должен быть выходной тензор (по умолчанию акой же, как и тип данных, выставленный по умолчанию в torch)
    """
    def __init__(self,
                 init_tensor_shape: Tuple[int],
                 seq_dim: int,
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(init_tensor_shape, device, dtype)
        
        self.seq_dim = seq_dim
        self.k = init_tensor_shape[1]
        
    def forward(self, *args, **kwargs) -> Tensor:
        output = super().forward()
        self.k += 1
        self.tensor_shape = self.tensor_shape[:self.seq_dim]\
            + (self.k,) + self.tensor_shape[self.seq_dim + 1:]
            
        return output
