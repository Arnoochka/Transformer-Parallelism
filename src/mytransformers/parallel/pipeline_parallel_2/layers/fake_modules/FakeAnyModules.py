from .FakeModule import FakeModule
from typing import Optional, Tuple, Any, Union, List
from .FakeTensorModules import FakeTupleTensorModule
import torch
from torch import Tensor
        
        
class FakeAnyModule(FakeModule):
    def __init__(self,
                 device: torch.device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(dtype, device)
        self.data: Any = None
    
    def forward(self, *args, **kwargs) -> Any:
        return self.data
    
    def set_gen_args(self, data: Any) -> None:
        self.data = data
        
class FakeTupleAnyTensorModule(FakeAnyModule):
    
    def __init__(self,
                 device,
                 dtype: Optional[torch.dtype] = None):
        super().__init__(device, dtype)
        self.tensor_idxs: List[int] = None 
        self.tensor_generator = FakeTupleTensorModule(device, dtype)
    
    def forward(self, *args, **kwargs) -> Tuple[Union[Tensor, Any]]:
        tensor_output = self.tensor_generator()
        output = self.data.copy()
        for idx, tensor_idx in enumerate(self.tensor_idxs):
            output[tensor_idx] = tensor_output[idx]
            
        return tuple(output)
    
    def set_gen_args(self, data: List[Union[Tuple[int], Any]], tensor_idxs: List[int]) -> None:
        self.tensor_idxs = tensor_idxs
        self.tensor_generator.set_gen_args([data[idx] for idx in tensor_idxs])
        super().set_gen_args(data)
            