import torch
import torch.distributed as dist
from torch import Tensor
from .MoeExperts import MoeExperts
from torch.distributed import ProcessGroup
from mytransformers.parallel.ParallelModule import ParallelModule
from .MoeExperts import MoeExperts
from typing import Callable, Optional
import torch.distributed as dist
from mytransformers.parallel.moe_parallel.moe_pipeline.pipeline.Scheduler import BaseScheduler

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
        
    @torch.no_grad()
    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        top_k_index, top_k_weights = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states
    
    def reset(self) -> None:
        self.thread_idx = 0
        self.experts.reset()
    
    
class MoeSparseBlockDPModule(MoeSparseBlockModule):
    def __init__(self,
                 experts: MoeExperts,
                 gate: Callable,
                 moe_group: ProcessGroup,
                 main_rank: int,
                 scheduler: BaseScheduler):
        super().__init__(experts, gate, moe_group, main_rank)
        self.register_buffer('buffer', None)
        self.scheduler = scheduler
        self.thread_idx = 0
     
    @torch.no_grad()   
    def forward(self, hidden_states: Optional[Tensor]) -> Tensor:
        rank = dist.get_rank()
        if rank == self.main_rank:
            splitted_hidden_states = list(torch.split(hidden_states,
                                                      hidden_states.size(0) // dist.get_world_size(self.moe_group),
                                                      dim=0))
        else: 
            splitted_hidden_states = None
            
        self.scheduler.transfer(op=dist.scatter,
                                op_info=self.thread_idx,
                                tensor=self.buffer,
                                scatter_list=splitted_hidden_states,
                                src=self.main_rank,
                                group=self.moe_group)
        
        output = super().forward(self.buffer)
        
        self.scheduler.transfer(op=dist.gather,
                        op_info=self.thread_idx,
                        tensor=output,
                        gather_list_list=splitted_hidden_states,
                        dst=self.main_rank,
                        group=self.moe_group)
        
        self.thread_idx += 1
        
        if self.main_rank == rank:
            return torch.cat(splitted_hidden_states, dim=0)
        else: 
            return None
    
    
class MoeSparseBlockPipeModule(MoeSparseBlockModule):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return super().forward(hidden_states)
        