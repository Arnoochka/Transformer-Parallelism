from typing import List
import torch
from torch.nn import Module
import torch.distributed as dist
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers.fake_modules import FakeModule, FakeTensorModule
from mytransformers.parallel.moe_parallel.layers import MoeComputeLayer, MoeFakeLayer, MoeSparseBlockModule

class MoeLayerGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: Module,
                moe: MoeSparseBlockModule,
                group_ranks: List[int],
                fake_module: FakeModule,
                device: torch.device) -> MoeComputeLayer:
        rank = dist.get_rank()
        if rank in group_ranks:
            return MoeComputeLayer(module, moe)
        else:
            return MoeFakeLayer(fake_module, moe, FakeTensorModule(device))
