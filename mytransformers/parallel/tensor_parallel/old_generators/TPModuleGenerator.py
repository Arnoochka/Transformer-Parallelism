from torch.distributed import ProcessGroup
from torch.nn import Module
import torch
from mytransformers.parallel.tensor_parallel.tp_layers import TPModule
    
class TPModuleGenerator:
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPModule:
        return TPModule(tp_group)