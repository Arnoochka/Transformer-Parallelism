from torch.distributed import ProcessGroup
from torch.nn import Module
import torch
from .parallel_layers import TensorParallelModule
    
class TensorParallelModuleGenerator(Module):
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TensorParallelModule:
        return TensorParallelModule(tp_group)