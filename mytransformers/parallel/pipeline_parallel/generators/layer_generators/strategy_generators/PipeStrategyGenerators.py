import torch
from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Tuple
from mytransformers.parallel.pipeline_parallel.layers import (PipeModule,
                                                              PipeDummyModule,
                                                              PipeRole,
                                                              PipeLeaderStrategyModule,
                                                              PipeLeaderTupleStrategyModule)
from mytransformers.parallel.pipeline_parallel.fake_output_generators import FakeGenerator
        
    

class LeaderStrategyGenerator:
    tensor_dim: int = 3
    def __new__(cls,
                role: PipeRole,
                module: Module,
                group_info: Tuple[ProcessGroup, List[int]],
                next_role: PipeRole,
                next_module: Module,
                next_group_info: Tuple[ProcessGroup, List[int]],
                fake_generator: FakeGenerator,
                leader_rank: int) -> PipeModule:
        rank = dist.get_rank()
        group, ranks = group_info
        next_group, next_ranks = next_group_info
        if rank in ranks:
            return PipeLeaderStrategyModule(role,
                                            module,
                                            group,
                                            next_group,
                                            cls.tensor_dim,
                                            leader_rank)
        elif rank in next_ranks:
            return PipeLeaderStrategyModule(next_role,
                                            next_module,
                                            group,
                                            next_group,
                                            cls.tensor_dim,
                                            leader_rank)
        else:
            return PipeDummyModule(fake_generator)
        
class LeaderTupleStrategyGenerator(LeaderStrategyGenerator):
    tensor_dim: int = 3
    def __new__(cls,
                role: PipeRole,
                module: Module,
                group_info: Tuple[ProcessGroup, List[int]],
                next_role: PipeRole,
                next_module: Module,
                next_group_info: Tuple[ProcessGroup, List[int]],
                fake_generator: FakeGenerator,
                leader_rank: int) -> PipeModule:
        rank = dist.get_rank()
        group, ranks = group_info
        next_group, next_ranks = next_group_info
        if rank in ranks:
            return PipeLeaderTupleStrategyModule(role,
                                                 module,
                                                 group,
                                                 next_group,
                                                 cls.tensor_dim,
                                                 leader_rank)
        elif rank in next_ranks:
            return PipeLeaderTupleStrategyModule(next_role,
                                                 next_module,
                                                 group,
                                                 next_group,
                                                 cls.tensor_dim,
                                                 leader_rank)
        else:
            return PipeDummyModule(fake_generator)