from torch.nn import Module
import torch
from .TPLinearLayersGenerator import(
    TPColumnLinearGenerator,
    TPRowLinearGenerator)
from torch.distributed import ProcessGroup
from .TPModuleGenerator import TPModuleGenerator

class TPFeedForwardGenerator(TPModuleGenerator):
    wi_gen = TPColumnLinearGenerator
    wo_gen = TPRowLinearGenerator 
    @torch.no_grad() 
    def __new__(cls,
                module: Module,
                tp_group: ProcessGroup) -> Module:
        cls.wi_gen.use_all_gather = False
        cls.wo_gen.use_all_reduce = True
        module.wi = cls.wi_gen(module.wi, tp_group)
        module.wo = cls.wo_gen(module.wo, tp_group)
        
        return module