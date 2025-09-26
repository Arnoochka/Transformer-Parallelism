from .PipeModule import PipeModule, PipeRole
from typing import Callable
from torch.nn import Module
from torch import Tensor

class PipeComputeModule(PipeModule):
    def __init__(self,
                 role: PipeRole,
                 module: Module):
        super().__init__(role, module)


        
    