from torch.distributed import ProcessGroup
from torch.nn import Module
from torch import Tensor

class ParallelModule(Module):
    def __init__(self, tp_group: ProcessGroup):
        super().__init__()
        self.tp_group = tp_group
        
    def forward(self, x: Tensor) -> Tensor:
        return x