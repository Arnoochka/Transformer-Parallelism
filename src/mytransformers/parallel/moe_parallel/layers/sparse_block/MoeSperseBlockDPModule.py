import torch
import torch.distributed as dist
from torch import Tensor
from mytransformers.parallel.moe_parallel.layers.experts import MoeExperts
from torch.distributed import ProcessGroup
from mytransformers.parallel.ParallelModule import ParallelModule
from typing import Callable, Optional, List, Any
import torch.distributed as dist
from mytransformers.parallel.moe_parallel.pipeline.Scheduler import BaseScheduler 
from .MoESparseBlockModule import MoeSparseBlockModule
from mytransformers.parallel.pipeline_parallel.layers import FakeModule

class MoeSparseBlockDPModule(MoeSparseBlockModule):
    def __init__(self,
                 experts: MoeExperts,
                 gate: Callable,
                 moe_group: ProcessGroup,
                 main_rank: int,
                 dim_buffer: Tensor,
                 scheduler: BaseScheduler):
        super().__init__(experts, gate, moe_group, main_rank, scheduler)
        self.register_buffer('dim_buffer', dim_buffer)
        
    @torch.no_grad()
    def forward(self, hidden_states: Tensor) -> Tensor:
        splitted_hidden_states = list(torch.split(hidden_states,
                                                  hidden_states.size(0) // dist.get_world_size(self.moe_group),
                                                  dim=0))
        hidden_states.shape
        self.dim_buffer.copy_(torch.tensor(splitted_hidden_states[0].size()))
        splitted_hidden_states = self.distributed_forward(splitted_hidden_states)
        
        return torch.cat(splitted_hidden_states, dim=0)
        
        
    @torch.no_grad()    
    def distributed_forward(self, splitted_hidden_states: List[Tensor]) -> List[Tensor]:
        self.scheduler.transfer(op=dist.broadcast,
                                op_info=self.thread_idx,
                                op_name="broadcast",
                                tensor=self.dim_buffer,
                                src=self.main_rank,
                                group=self.moe_group)
        
        buffer = torch.empty(self.dim_buffer.tolist(), device=torch.cuda.current_device())
        
        self.scheduler.transfer(op=dist.scatter,
                        op_info=self.thread_idx,
                        op_name="scatter",
                        tensor=buffer,
                        scatter_list=splitted_hidden_states,
                        src=self.main_rank,
                        group=self.moe_group)
        
        output = self.compute(buffer)
        
        self.scheduler.transfer(op=dist.gather,
                        op_info=self.thread_idx,
                        op_name="gather",
                        tensor=output,
                        gather_list=splitted_hidden_states,
                        dst=self.main_rank,
                        group=self.moe_group)
        
        self.thread_idx += 1
        
        return splitted_hidden_states
    
class MoeSparseBlocFakekDPModule(MoeSparseBlockDPModule):
    def __init__(self,
                 experts: MoeExperts,
                 gate: Callable,
                 moe_group: ProcessGroup,
                 main_rank: int,
                 dim_buffer: Tensor,
                 scheduler: BaseScheduler,
                 fake_module: FakeModule):
        super().__init__(experts, gate, moe_group, main_rank, dim_buffer, scheduler) 
        self.fake_module = fake_module
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Any:
        self.distributed_forward(None)
        return self.fake_module(*args, **kwargs)
