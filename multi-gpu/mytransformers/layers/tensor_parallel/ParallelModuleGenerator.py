from torch.distributed import ProcessGroup
from torch.nn import Module
from .layers import TensorParallelModule
    
class TensorParallelModuleGenerator(Module):
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TensorParallelModule:
        return TensorParallelModule(tp_group)