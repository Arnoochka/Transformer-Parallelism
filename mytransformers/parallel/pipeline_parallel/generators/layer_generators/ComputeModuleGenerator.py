from typing import List
from torch.nn import Module
import torch.distributed as dist
from mytransformers.parallel.pipeline_parallel.layers import (PipeFakeModule, PipeRole,
                                                              PipeComputeModule,
                                                              PipeModule,
                                                              FakeModule)

class ComputeModuleGenerator:
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
