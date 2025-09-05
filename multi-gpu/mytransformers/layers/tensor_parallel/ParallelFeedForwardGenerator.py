from torch.nn import Module
import torch
from .ParallelLinearLayersGenerator import(
    ColumnParallelLinearGenerator,
    RowParallelLinearGenerator)
from torch.distributed import ProcessGroup
from .ParallelModuleGenerator import TensorParallelModuleGenerator
from .parallel_layers import ParallelFeedForward
    
class ParallelFeedForwardGenerator(TensorParallelModuleGenerator):
    @torch.no_grad() 
    def __new__(cls,
                module: Module,
                tp_group: ProcessGroup) -> ParallelFeedForward:
        ColumnParallelLinearGenerator.use_all_gather = False
        RowParallelLinearGenerator.use_all_reduce = True
        wi = ColumnParallelLinearGenerator(module.wi, tp_group)
        wo = RowParallelLinearGenerator(module.wo, tp_group)
        
        return ParallelFeedForward(wi, wo, tp_group)