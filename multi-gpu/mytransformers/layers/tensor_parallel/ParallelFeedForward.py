import torch.nn as nn
from torch.nn import Module
from torch import Tensor
from typing import Optional
from .ParallelLinearLayers import (ColumnParallelLinearGenerator,
                                   RowParallelLinearGenerator,
                                   ParallelLinear)
from torch.distributed import ProcessGroup
from .ParallelModule import TensorParallelModule, TensorParallelModuleGenerator

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
    
class ParallelFeedForwardGenerator(TensorParallelModuleGenerator):
    config = None    
    def __new__(cls,
                module: Module,
                tp_group: ProcessGroup) -> ParallelFeedForward:
        ColumnParallelLinearGenerator.use_all_gather = False
        RowParallelLinearGenerator.use_all_reduce = True
        wi = ColumnParallelLinearGenerator(module.wi)
        wo = RowParallelLinearGenerator(module.wo)
        
        return ParallelFeedForward(cls.config, wi, wo, tp_group)