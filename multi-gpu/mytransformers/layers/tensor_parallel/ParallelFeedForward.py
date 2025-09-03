import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
import numpy as np
from .ParallelLinearLayers import ColumnParallelLinearGenerator, RowParallelLinearGenerator
from torch.distributed import ProcessGroup
from .ParallelModule import TensorParallelModule, TensorParallelModuleGenerator

class ParallelFeedForward(TensorParallelModule):
    def __init__(self,
                 config,
                 wi: Module,
                 wo: Module,
                 tp_group: ProcessGroup):
        super().__init__(tp_group)
        self.func = nn.ReLU()
        self.wi = ColumnParallelLinearGenerator(wi, tp_group, use_all_gather = False)
        self.wo = RowParallelLinearGenerator(wo, tp_group, use_all_reduce = True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> Tensor:
        x = self.func(self.wi(x))
        logits = self.wo(self.dropout(x))
        return logits
    
class ParallelFeedForwardGenerator(TensorParallelModuleGenerator):
    def __new__(cls, module: Module, tp_group: ProcessGroup, config) -> ParallelFeedForward:
        return ParallelFeedForward(config, module.wi, module.wo, tp_group)