from torch.distributed import ProcessGroup
from torch.nn import Module
import torch
    
class TPModuleGenerator:
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        return module