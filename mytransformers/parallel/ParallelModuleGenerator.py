from mytransformers.parallel import ParallelModule
from torch.nn import Module
import torch
    
class ParallelModuleGenerator(Module):
    @torch.no_grad()
    def __new__(cls, module: Module) -> ParallelModule:
        return ParallelModule()