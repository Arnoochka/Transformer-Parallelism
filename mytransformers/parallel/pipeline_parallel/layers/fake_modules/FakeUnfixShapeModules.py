from typing import List, Tuple
import torch
from torch import Tensor
from .FakeModule import FakeModule

class FakeUnfixShapeTensorModule(FakeModule):
    tensor_shape: Tuple[int]     
    def forward(self, *args, **kwargs) -> Tensor:
        return self.get_tensor(self, *args, **kwargs)
    
    def get_tensor(self, *args, **kwargs) -> Tensor:
        return torch.zeros(self.tensor_shape, device=self.device, dtype=self.dtype)
    
    
class FakeUnfixShapeTupleModule(FakeUnfixShapeTensorModule):
    tensor_shapes: List[Tuple[int]]
    def forward(self, *args, **kwargs) -> Tuple[Tensor]:
        outputs = [self.get_tensor(tensor_shape)
                   for tensor_shape in self.tensor_shapes]
        return tuple(outputs)
        

