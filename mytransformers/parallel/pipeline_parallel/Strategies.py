from torch.nn import Module
from torch.distributed import ProcessGroup
import torch.distributed as dist
from mytransformers.parallel.ParallelModule import ParallelModule
from typing import Optional, Callable
from torch import Tensor

class Strategy(ParallelModule):
    def __init__(self,
                 stage_group: ProcessGroup,
                 next_stage_group: ProcessGroup):
        super().__init__()
        self.stage_group = stage_group
        self.next_stage_group = next_stage_group
        
    def __call__(self, x: Tensor) -> None:
        return x
    
    
class LeaderStrategy(Strategy):
    def __init__(self,
                 stage_group: ProcessGroup,
                 next_stage_group: ProcessGroup):
        super().__init__(stage_group, next_stage_group)
        
    def __call__(self, x: Tensor) -> None:
        rank = dist.get_process_group_ranks(self.stage_group)[0]
        next_rank = dist.get_process_group_ranks(self.stage_group)[0]
        if dist.get_rank() == rank:
            dist.isend(x, next_rank)
        elif dist.get_rank() == next_rank:
            dist.()
        
        