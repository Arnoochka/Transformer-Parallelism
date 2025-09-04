import torch.nn as nn
from torch.nn import Module
from torch import Tensor
from .ParallelLinearLayersGenerator import (ColumnParallelLinearGenerator,
                                   RowParallelLinearGenerator)
from torch.distributed import ProcessGroup
from .ParallelModuleGenerator import TensorParallelModule, TensorParallelModuleGenerator
from .layers import ParallelFeedForward
    
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