from typing import List
from torch.nn import Module
import torch.distributed as dist
from mytransformers.parallel.pipeline_parallel.layers import (PipeFakeModule, PipeRole,
                                                              PipeComputeModule,
                                                              PipeModule,
                                                              FakeModule)

class ComputeModuleGenerator:
    """
    генератор, для вычислительного модуля. На нужно ппроцесса сзодает вычислительный модуль, на остальных - фейковый
    
    Args:
        role (PipeRole): роль модуля (compute или recvAndCompute)
        module (Module): модуль, который будет производить вычисления
        group_ranks (List[int]): группа рангов, у которых должен быть вычислительный модуль
        fake_module: (FakeModule): тип "фейкового" модуля для остальных рангов
    """
    def __new__(cls,
                role: PipeRole,
                module: Module,
                group_ranks: List[int],
                fake_module: FakeModule) -> PipeModule:
        rank = dist.get_rank()
        if rank in group_ranks:
            return PipeComputeModule(role, module)
        else:
            return PipeFakeModule(fake_module)
