import torch.nn as nn
import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from .ParallelModule import TensorParallelModule

class ParallelFeedForward(TensorParallelModule):
    def __init__(self,
                 wi: TensorParallelModule,
                 wo: TensorParallelModule,
                 tp_group: ProcessGroup):
        super().__init__(tp_group)
        self.func = nn.ReLU()
        self.wi = wi
        self.wo = wo

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = self.func(self.wi(x))
        logits = self.wo(x)
        return logits