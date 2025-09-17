from torch.distributed import ProcessGroup
from torch.nn import Module
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
import torch
    
class TPModuleGenerator(ParallelModuleGenerator):
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        return super().__new__(module, tp_group)