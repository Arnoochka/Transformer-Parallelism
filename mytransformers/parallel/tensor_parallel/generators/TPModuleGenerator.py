from torch.distributed import ProcessGroup
from torch.nn import Module
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.tensor_parallel.layers import TPModule
import warnings
import torch
    
class TPModuleGenerator(ParallelModuleGenerator):
    tp_group: ProcessGroup = None
    @torch.no_grad()
    def __new__(cls, module: Module, device: torch.device) -> TPModule:
        return super().__new__(module)
    
    @staticmethod
    def already_converted(module: Module) -> bool:
        """
        проверяет, является ли модуль уже Тензорным слоем. 
        Если да, то выкидывает warning
        """
        if isinstance(module, TPModule):
            warnings.warn(
                f"this module is already converted in TPModule: {type(module).__name__}",
                UserWarning,
                stacklevel=5)
            return True
        return False