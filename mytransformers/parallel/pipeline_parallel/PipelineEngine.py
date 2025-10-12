from torch.nn import Module
import torch
from torch import LongTensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .layers import PipeModule
from typing import Dict, Any

class PipelineEngine(Module):
    def __init__(self,
                 module: PipeModule):
        super().__init__()
        self.module = module
        
    def forward(self, module_kwargs: Dict) -> Any:
        return self.module(**module_kwargs)
    
    def generate(self):
        pass