import torch
from torch.distributed import ProcessGroup
from typing import List, Callable, Optional
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.moe_parallel.layers import MoeExperts
from .MoeExpertsGenerator import MoeExpertsModuleGenerator
from mytransformers.parallel.moe_parallel.layers import MoeSparseBlockDPModule, MoeSparseBlockModule
from mytransformers.parallel.moe_parallel.pipeline.Scheduler import BaseScheduler

class MoeSparseBlockDPModuleGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: Module,
                gate: Callable,
                k: int,
                main_rank: int,
                next_main_rank: int,
                replace_experts_layer: MoeExperts,
                expert_idxs: List[Tensor],
                moe_group: ProcessGroup,
                scheduler: BaseScheduler,
                device: torch.device,
                ) -> MoeExperts:
        experts = MoeExpertsModuleGenerator(module.experts,
                                            replace_experts_layer,
                                            expert_idxs,
                                            moe_group,
                                            scheduler,
                                            device)
        return MoeSparseBlockDPModule(experts, gate, k, moe_group, main_rank, next_main_rank, scheduler).to(device)
    
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
        # TODO: нужно править
        experts = MoeExpertsModuleGenerator(module.experts,
                                            replace_experts_layer,
                                            expert_idxs,
                                            moe_group,
                                            scheduler,
                                            device,
                                            main_rank=main_rank)
        return MoeSparseBlockModule(experts, gate, moe_group, main_rank).to(device)
        
        
    