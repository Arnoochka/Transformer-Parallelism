from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import (PipeRole, PipeComputeModule, PipeDummyModule, PipeModule)
from typing import Callable, List
from torch.nn import Module
import torch.distributed as dist
import torch

class PipeComputeModuleGenerator(ParallelModuleGenerator):
    role: PipeRole = None
    neighbor_role: PipeRole = None
    neighbor_module: Callable = None
    group_ranks: List[int] = None
    next_group_ranks: List[int] = None

    def __new__(cls, module: Module, device: torch.device) -> PipeModule:
        rank = dist.get_rank()
        if rank in cls.group_ranks:
            return PipeComputeModule(cls.role, module).to(device)
        elif rank in cls.next_group_ranks:
            return PipeComputeModule(cls.neighbor_role, cls.neighbor_module).to(device)
        else:
            return PipeDummyModule(device).to(device)
