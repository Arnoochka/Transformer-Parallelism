from typing import List
from torch.nn import Module
import torch.distributed as dist
import torch
from mytransformers.parallel.pipeline_parallel.layers import (PipeRole,
                                                              PipeComputeModule, 
                                                              PipeDummyModule,
                                                              PipeModule)

class ComputeModuleGenerator:
    def __new__(cls,
                role: PipeRole,
                module: Module,
                group_ranks: List[int],
                device: torch.device) -> PipeModule:
        rank = dist.get_rank()
        if rank in group_ranks:
            return PipeComputeModule(role, module)
        else:
            return PipeDummyModule(device)
