from torch.distributed import ProcessGroup
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from .Strategies import Strategy, LeaderStrategy, CommRole
import torch.distributed as dist
from typing import Tuple
from torch import Tensor
from enum import Enum
import torch

class StrategyGenerator(ParallelModuleGenerator):
    def __new__(cls,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup):
        super().__init__()
        return Strategy(CommRole.none,
                        send_group,
                        recv_group)

    
    
class LeaderStrategyGenerator(StrategyGenerator):
    leader_rank: int = 0
    def __new__(cls,
                 comm_role: CommRole,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup) -> LeaderStrategy:
        return LeaderStrategy(comm_role,
                              send_group,
                              recv_group,
                              cls.leader_rank)
                
    