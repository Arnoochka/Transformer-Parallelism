from typing import List
from torch.nn import Module
import torch.distributed as dist
from mytransformers.parallel.pipeline_parallel.layers import (PipeRole,
                                                              PipeComputeModule,
                                                              PipeDummyModule,
                                                              PipeModule)
from mytransformers.parallel.pipeline_parallel.fake_output_generators import FakeGenerator

class ComputeModuleGenerator:
    def __new__(cls,
                role: PipeRole,
                module: Module,
                group_ranks: List[int],
                fake_generator: FakeGenerator) -> PipeModule:
        rank = dist.get_rank()
        if rank in group_ranks:
            return PipeComputeModule(role, module)
        else:
            return PipeDummyModule(fake_generator)
