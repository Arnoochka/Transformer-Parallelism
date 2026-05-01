from typing import Tuple, Any, Callable, Optional
import torch
from torch.nn import Module
from mytransformers.parallel.ParallelModule import ParallelModule
from mytransformers.parallel.pipeline_parallel.layers.fake_modules import FakeModule, FakeTensorModule
from .sparse_block import MoeSparseBlockModule


class MoeComputeLayer(ParallelModule):
    def __init__(self,
                 module: Module,
                 moe: MoeSparseBlockModule):
        super().__init__()
        self.module = module
        self.moe = moe
        self.notify_func: Optional[Callable] = None
        
    def __call__(self, *args, **kwds) -> Any:
        output = super().__call__(*args, **kwds)
        self.notify()
        return output
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Any:
        output = self.module(*args, **kwargs)
        return output
        
    def reset(self) -> None:
        self.moe.reset()
        
    def set_notify(self, notify_func: Callable) -> None:
        self.notify_func = notify_func
        
    def notify(self) -> None:
        self.moe.update_thread_idx()
        if self.notify_func is not None:
            self.notify_func()
        
        
class MoeFakeLayer(MoeComputeLayer):
    def __init__(self,
                 fake_module: FakeModule, 
                 moe: MoeSparseBlockModule,
                 fake_hidden_module: FakeTensorModule):
        super().__init__(fake_module, moe)
        self.fake_hidden_module = fake_hidden_module
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Any:
        self.moe(self.fake_hidden_module())
        return super().forward(*args, **kwargs)
        
    
    def set_gen_args(self, input_shape: Tuple[int], *fake_layer_args) -> None:
        self.fake_hidden_module.set_gen_args(input_shape)
        self.module.set_gen_args(*fake_layer_args)
        
        