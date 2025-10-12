from .FakeGenerator import FakeGenerator
from typing import Optional, Tuple
import torch
from torch import Tensor

class FakeTensorGenerator(FakeGenerator):
    def __init__(self,
                 tensor_shape: Tuple[int],
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(device, dtype)
        self.tensor_shape = tensor_shape
        
    def generate(self, *args, **kwargs) -> Tensor:
        return torch.zeros(self.tensor_shape, device=self.device, dtype=self.dtype)
        
        
class FakeSeqGenerator(FakeTensorGenerator):
    def __init__(self,
                 init_tensor_shape: Tuple[int],
                 seq_dim: int,
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(init_tensor_shape, device, dtype)
        
        self.seq_dim = seq_dim
        self.k = init_tensor_shape[1]
        
    def generate(self, *args, **kwargs) -> Tensor:
        output = super().generate()
        self.k += 1
        self.tensor_shape = self.tensor_shape[:self.seq_dim]\
            + (self.k,) + self.tensor_shape[self.seq_dim + 1:]
            
        return output
        
    
class FakeUnfixShapeTensorGenerator(FakeTensorGenerator):
    def set_tensor_shape(self, tensor_shape: Tuple[int]) -> None:
        self.tensor_shape = tensor_shape
