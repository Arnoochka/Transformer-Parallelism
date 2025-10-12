from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Tuple, Dict
from mytransformers.parallel.pipeline_parallel.layers import (PipeModule,
                                                              PipeDummyModule,
                                                              PipeRole,
                                                              PipeStrategyModule)
from mytransformers.parallel.pipeline_parallel.fake_output_generators import FakeGenerator
        
    

class StrategyModuleGenerator:
    tensor_dim: int = 3
    strategy_module: PipeStrategyModule = PipeStrategyModule
    def __new__(cls,
                role: PipeRole,
                module: Module,
                group_info: Tuple[ProcessGroup, List[int]],
                next_role: PipeRole,
                next_module: Module,
                next_group_info: Tuple[ProcessGroup, List[int]],
                fake_generator: FakeGenerator,
                strategy_kwargs: Dict = {}) -> PipeModule:
        rank = dist.get_rank()
        group, ranks = group_info
        next_group, next_ranks = next_group_info
        if rank in ranks:
            return cls.strategy_module(role,
                                       module,
                                       group,
                                       next_group,
                                       cls.tensor_dim,
                                       **strategy_kwargs)
        elif rank in next_ranks:
            return cls.strategy_module(next_role,
                                       next_module,
                                       group,
                                       next_group,
                                       cls.tensor_dim,
                                       **strategy_kwargs)
        else:
            return PipeDummyModule(fake_generator)