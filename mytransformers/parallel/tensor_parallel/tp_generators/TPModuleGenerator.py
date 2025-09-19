from torch.distributed import ProcessGroup
from torch.nn import Module
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.tensor_parallel.tp_layers import TPModule
import warnings
import torch
    
class TPModuleGenerator(ParallelModuleGenerator):
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPModule:
        return super().__new__(module, tp_group)
    
    @staticmethod
    def already_conferted(module: Module) -> bool:
        if isinstance(module, TPModule):
            warnings.warn(
                f"this module is already converted in TPModule: {type(module).__name__}",
                UserWarning,
                stacklevel=5)
            return True
        return False