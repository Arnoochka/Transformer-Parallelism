from torch.distributed import ProcessGroup
from mytransformers.parallel import ParallelModule
from torch.nn import Module
import torch
    
class ParallelModuleGenerator(Module):
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelModule:
        return ParallelModule(tp_group)