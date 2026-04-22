import torch
import torch.distributed as dist
from torch import Tensor
from ..experts.MoeExperts import MoeExperts
from torch.distributed import ProcessGroup
from mytransformers.parallel.ParallelModule import ParallelModule
from ..experts.MoeExperts import MoeExperts
from typing import Callable, Optional
from mytransformers.parallel.moe_parallel.pipeline.Scheduler import BaseScheduler

class MoeSparseBlockModule(ParallelModule):
    def __init__(self,
                 experts: MoeExperts,
                 gate: Callable,
                 moe_group: ProcessGroup,
                 main_rank: int,
                 scheduler: Optional[BaseScheduler]):
        super().__init__()
        self.experts = experts
        self.gate = gate
        self.moe_group = moe_group
        self.main_rank = main_rank
        self.scheduler = scheduler
        self.thread_idx = 0
        
    def forward(self, *args, **kwargs) -> Tensor:
        return super().forward(*args, **kwargs)
        
    @torch.no_grad()
    def compute(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        top_k_index, top_k_weights = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states
    
    def reset(self) -> None:
        self.thread_idx = 0
        self.experts.reset()
        