from torch.distributed import ProcessGroup
from mytransformers.parallel.ParallelModule import ParallelModule
import torch.distributed as dist
from torch import Tensor
import torch

class TPModule(ParallelModule):
    def __init__(self, tp_group: ProcessGroup):
        super().__init__(tp_group)
        
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)