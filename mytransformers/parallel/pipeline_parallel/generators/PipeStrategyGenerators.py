import torch
from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from mytransformers.parallel import ParallelModuleGenerator
from typing import List, Tuple
from mytransformers.parallel.pipeline_parallel.layers import (PipeStrategy,
                                                              PipeRole,
                                                              PipeDummyModule,
                                                              PipeModule,
                                                              PipeLeaderStrategy,
                                                              PipeBroadcastLeaderStrategy)

class PipeStrategyGenerator(ParallelModuleGenerator):
    role: PipeRole = PipeRole.none
    neighbor_role: PipeRole = None
    neighbor_module: Module = None
    group_info: Tuple[ProcessGroup, List[int]] = None
    next_group_info: Tuple[ProcessGroup, List[int]] = None
    tensor_dim: int = 3
    def __new__(cls, module: Module, device: torch.device) -> PipeStrategy:
        rank = dist.get_rank()
        group, ranks = cls.group_info
        next_group, next_ranks = cls.next_group_info
        return PipeStrategy(cls.role,
                            module,
                            group,
                            next_group,
                            cls.tensor_dim).to(device)
    

class PipeLeaderStrategyGenerator(PipeStrategyGenerator):
    role: PipeRole = PipeRole.none
    neighbor_role: PipeRole = None
    neighbor_module: Module = None
    group_info: Tuple[ProcessGroup, List[int]] = None
    next_group_info: Tuple[ProcessGroup, List[int]] = None
    tensor_dim: int = 3
    leader_rank: int = 0

    def __new__(cls, module: Module, device: torch.device) -> PipeModule:
        rank = dist.get_rank()
        group, ranks = cls.group_info
        next_group, next_ranks = cls.next_group_info
        if rank in ranks:
            return PipeLeaderStrategy(cls.role,
                                      module,
                                      group,
                                      next_group,
                                      cls.tensor_dim,
                                      cls.leader_rank).to(device)
        elif rank in next_ranks:
            return PipeLeaderStrategy(cls.neighbor_role,
                                      cls.neighbor_module,
                                      group,
                                      next_group,
                                      cls.tensor_dim,
                                      cls.leader_rank).to(device)
        else:
            return PipeDummyModule(device).to(device)
        
        
class PipeBroadcastLeaderStrategyGenerator(PipeLeaderStrategyGenerator):
    role: PipeRole = PipeRole.none
    neighbor_role: PipeRole = None
    neighbor_module: Module = None
    group_info: Tuple[ProcessGroup, List[int]] = None
    tensor_dim: int = 3
    leader_rank: int = 0

    def __new__(cls, module: Module, device: torch.device) -> PipeModule:
        rank = dist.get_rank()
        group, ranks = cls.group_info
        if rank in ranks:
            return PipeBroadcastLeaderStrategy(cls.role,
                                      module,
                                      group,
                                      None,
                                      cls.tensor_dim,
                                      cls.leader_rank).to(device)
        else:
            return PipeBroadcastLeaderStrategy(cls.neighbor_role,
                                      cls.neighbor_module,
                                      group,
                                      None,
                                      cls.tensor_dim,
                                      cls.leader_rank).to(device)