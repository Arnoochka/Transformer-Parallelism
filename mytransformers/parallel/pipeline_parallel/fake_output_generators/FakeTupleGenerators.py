from typing import Optional, List, Tuple
import torch
from torch import Tensor
from .FakeGenerator import FakeGenerator

class FakeTupleTensorGenerator(FakeGenerator):
    def __init__(self,
                 tensor_shapes: List[Tuple[int]],
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(dtype)
        self.tensor_shapes = tensor_shapes
        self.device = device
    
    def generate(self, *args, **kwargs) -> Tuple[Tensor]:
        outputs = [self.get_tensor(tensor_shape)
                   for tensor_shape in self.tensor_shapes]
        return tuple(outputs)
    
    def get_tensor(self, tensor_shape: Tuple[int]) -> Tensor:
        return torch.zeros(tensor_shape, device=self.device, dtype=self.dtype)
    
    
class FakeTupleOptionalTensorGenerator(FakeTupleTensorGenerator):
    def __init__(self,
                 tensor_shapes: List[Optional[Tuple[int]]],
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(tensor_shapes, device, dtype)
        
    def generate(self, *args, **kwargs) -> Tuple[Optional[Tensor]]:
        outputs = [self.get_tensor(tensor_shape) if tensor_shape is not None else None
                   for tensor_shape in self.tensor_shapes]
        return tuple(outputs)
                
        