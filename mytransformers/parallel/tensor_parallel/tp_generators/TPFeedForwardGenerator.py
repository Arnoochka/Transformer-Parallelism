from torch.nn import Module
import torch
from .TPLinearLayersGenerator import(
    TPColumnLinearGenerator,
    TPRowLinearGenerator)
from torch.distributed import ProcessGroup
from .TPModuleGenerator import TPModuleGenerator
from mytransformers.parallel.tensor_parallel.tp_layers import TPFeedForward

class TPFeedForwardGenerator(TPModuleGenerator):
    @torch.no_grad() 
    def __new__(cls,
                module: Module,
                tp_group: ProcessGroup) -> TPFeedForward:
        TPColumnLinearGenerator.use_all_gather = False
        TPRowLinearGenerator.use_all_reduce = True
        wi = TPColumnLinearGenerator(module.wi, tp_group)
        wo = TPRowLinearGenerator(module.wo, tp_group)
        
        return TPFeedForward(wi, wo, tp_group)