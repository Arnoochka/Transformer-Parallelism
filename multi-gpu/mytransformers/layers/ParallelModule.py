from torch.distributed import ProcessGroup
from torch.nn import Module

class TensorParallelModule(Module):
    def __init__(self, tp_group: ProcessGroup):
        super().__init__()
        self.tp_group = tp_group
        
    @staticmethod
    def from_no_parallel(module: Module, tp_group: ProcessGroup) -> "TensorParallelModule":
        pass