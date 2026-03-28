import torch
import torch.distributed as dist
from torch import Tensor
from .MoeExperts import MoeExperts
from torch.distributed import ProcessGroup
from mytransformers.parallel.ParallelModule import ParallelModule
from .MoeExperts import MoeExperts
from typing import Callable, Optional
import torch.distributed as dist
from mytransformers.utils import Logger

class MoeSparseBlockModule(ParallelModule):
    def __init__(self,
                 experts: MoeExperts,
                 gate: Callable,
                 moe_group: ProcessGroup,
                 main_rank: int):
        super().__init__()
        self.experts = experts
        self.gate = gate
        self.moe_group = moe_group
        self.main_rank = main_rank
        
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        top_k_index, top_k_weights = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states
    
    
class MoeSparseBlockDPModule(MoeSparseBlockModule):
    def __init__(self,
                 experts: MoeExperts,
                 gate: Callable,
                 moe_group: ProcessGroup,
                 main_rank: int,
                 buffer: Tensor):
        super().__init__(experts, gate, moe_group, main_rank)
        self.register_buffer('buffer', buffer)
        
    def forward(self, hidden_states: Optional[Tensor]) -> Tensor:
        rank = dist.get_rank()
        if rank == self.main_rank:
            splitted_hidden_states = list(torch.split(hidden_states,
                                                      hidden_states.size(0) // dist.get_world_size(self.moe_group),
                                                      dim=0))
        else: 
            splitted_hidden_states = None
        dist.scatter(self.buffer, splitted_hidden_states, src=self.main_rank, group=self.moe_group)
        
        output = super().forward(self.buffer)
        
        dist.gather(output, splitted_hidden_states, dst=self.main_rank, group=self.moe_group)
        if self.main_rank == rank:
            return torch.cat(splitted_hidden_states, dim=0)
        else: 
            return None
    
    
class MoeSparseBlockPipeModule(MoeSparseBlockModule):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return super().forward(hidden_states)
        