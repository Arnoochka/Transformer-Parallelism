import torch.nn as nn
import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from .TPModule import TPModule

class TPFeedForward(TPModule):
    def __init__(self,
                 wi: TPModule,
                 wo: TPModule,
                 tp_group: ProcessGroup):
        super().__init__(tp_group)
        self.func = nn.SELU()
        self.wi = wi
        self.wo = wo

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = self.func(self.wi(x))
        logits = self.wo(x)
        return logits