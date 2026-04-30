import torch
import torch.distributed as dist
from torch import Tensor
from mytransformers.parallel.moe_parallel.layers.experts import MoeExperts
from torch.distributed import ProcessGroup
from mytransformers.parallel.ParallelModule import ParallelModule
from typing import Callable, Optional
from mytransformers.parallel.moe_parallel.pipeline.Scheduler import BaseScheduler

class MoeSparseBlockModule(ParallelModule):
    def __init__(self,
                 experts: MoeExperts,
                 gate: Callable,
                 k: int,
                 moe_group: ProcessGroup,
                 main_rank: int,
                 next_main_rank: int,
                 scheduler: Optional[BaseScheduler]):
        super().__init__()
        self.experts = experts
        self.gate = gate
        self.k = k
        self.moe_group = moe_group
        self.main_rank = main_rank
        self.next_main_rank = next_main_rank
        self.scheduler = scheduler
        self.thread_idx = 0
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        return super().forward()
    
    def reset(self) -> None:
        self.thread_idx = 0
        self.experts.reset()
        
    def update_thread_idx(self) -> None:
        self.thread_idx += 1
        self.experts.update_thread_idx()
        