import torch
from torch.distributed import ProcessGroup
from typing import List, Callable, Optional
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.moe_parallel.layers import MoeExperts
from .MoeExpertsGenerator import MoeExpertsModuleGenerator
from mytransformers.parallel.moe_parallel.layers import MoeSparseBlockDPModule, MoeSparseBlocFakekDPModule, MoeSparseBlockModule
from mytransformers.parallel.moe_parallel.pipeline.Scheduler import BaseScheduler
from mytransformers.parallel.pipeline_parallel.layers import FakeModule

class MoeSparseBlockDPModuleGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: Module,
                gate: Callable,
                main_rank: int,
                replace_experts_layer: MoeExperts,
                expert_idxs: List[Tensor],
                moe_group: ProcessGroup,
                dim_buffer: Tensor,
                scheduler: BaseScheduler,
                device: torch.device,
                fake_module: Optional[FakeModule] = None
                ) -> MoeExperts:
        experts = MoeExpertsModuleGenerator(module.experts,
                                            replace_experts_layer,
                                            expert_idxs,
                                            moe_group,
                                            scheduler,
                                            device)
        if dist.get_rank() == main_rank:
                return MoeSparseBlockDPModule(experts, gate, moe_group, main_rank, dim_buffer, scheduler).to(device)
        else:
                return MoeSparseBlocFakekDPModule(experts, gate, moe_group, main_rank, dim_buffer, scheduler, fake_module).to(device)
    
class MoeSparseBlockPipeModuleGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: Module,
                gate: Callable,
                main_rank: int,
                replace_experts_layer: MoeExperts,
                expert_idxs: List[Tensor],
                moe_group: ProcessGroup,
                scheduler: BaseScheduler,
                device: torch.device) -> MoeExperts:
        experts = MoeExpertsModuleGenerator(module.experts,
                                            replace_experts_layer,
                                            expert_idxs,
                                            moe_group,
                                            scheduler,
                                            device,
                                            main_rank=main_rank)
        return MoeSparseBlockModule(experts, gate, moe_group, main_rank).to(device)
        
        
    