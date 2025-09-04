import torch.nn as nn
from torch import Tensor
from torch.distributed import ProcessGroup
from .ParallelModule import TensorParallelModule

class ParallelFeedForward(TensorParallelModule):
    def __init__(self,
                 config,
                 wi: TensorParallelModule,
                 wo: TensorParallelModule,
                 tp_group: ProcessGroup):
        super().__init__(tp_group)
        self.func = nn.ReLU()
        self.wi = wi
        self.wo = wo
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> Tensor:
        x = self.func(self.wi(x))
        logits = self.wo(self.dropout(x))
        return logits