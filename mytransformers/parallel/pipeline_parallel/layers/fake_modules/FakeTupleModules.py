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
                 tensor_shapes: List[Tuple[int]],
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(dtype)
        self.tensor_shapes = tensor_shapes
        self.device = device
    
    def forward(self, *args, **kwargs) -> Tuple[Tensor]:
        outputs = [self.get_tensor(tensor_shape)
                   for tensor_shape in self.tensor_shapes]
        return tuple(outputs)
    
    def get_tensor(self, tensor_shape: Tuple[int]) -> Tensor:
        return torch.zeros(tensor_shape, device=self.device, dtype=self.dtype)
    
    
class FakeTupleOptionalTensorModule(FakeTupleTensorModule):
    """
    генерирует кортеж фейковых тензоров необходимой размерности. Однако, некоторые значения в кортеже могут быть None
    
    Args:
        tensor_shape (List[Optional[Tuple[int]]]): начальные размеры выходных тензоров
        device (torch.device): устройство, на котором должен быть выходной тензор
        dtype (torch.dtype): тип данных, с которым должен быть выходной тензор (по умолчанию акой же, как и тип данных, выставленный по умолчанию в torch)
    """
    def __init__(self,
                 tensor_shapes: List[Optional[Tuple[int]]],
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(tensor_shapes, device, dtype)
        
    def forward(self, *args, **kwargs) -> Tuple[Optional[Tensor]]:
        outputs = [self.get_tensor(tensor_shape) if tensor_shape is not None else None
                   for tensor_shape in self.tensor_shapes]
        return tuple(outputs)
    
class FakeTupleSeqModule(FakeTupleTensorModule):
    """
    генерирует кортеж фейковых тензоров необходимой размерности. Однако, при каждой новой генерации увеличивает нужные размерности на 1
    
    Args:
        init_tensor_shape (Tuple[int]): начальные размеры выходных тензоров
        seq_dims (int): размерности, которые будут увиличиваться
        device (torch.device): устройство, на котором должен быть выходной тензор
        dtype (torch.dtype): тип данных, с которым должен быть выходной тензор (по умолчанию акой же, как и тип данных, выставленный по умолчанию в torch)
    """
    def __init__(self, 
                 init_tensor_shapes: List[Tuple[int]],
                 seq_dims: List[int],
                 device,
                 dtype = None):
        super().__init__(init_tensor_shapes, device, dtype)
        self.seq_dims = seq_dims
        self.ks = [init_tensor_shapes[idx][seq_dim]
                   for idx, seq_dim in enumerate(seq_dims)]
        
    def forward(self, *args, **kwargs):
        output  = super().forward(*args, **kwargs)
        for idx in range(len(self.tensor_shapes)):
            self.ks[idx] += 1
            self.tensor_shapes[idx] = self.tensor_shapes[idx][:self.seq_dims[idx]] + \
                (self.ks[idx], ) + self.tensor_shapes[idx][self.seq_dims[idx] + 1:]
                
        return output
        