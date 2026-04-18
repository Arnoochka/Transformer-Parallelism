import torch
from torch.distributed import ProcessGroup
from typing import List, Callable
import torch.distributed as dist
from torch import Tensor
from torch.nn import ModuleList, Module
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.moe_parallel.moe_pipeline.layers.moe_layers import MoeExperts
from .MoeExpertsGenerator import MoeExpertsModuleGenerator
from mytransformers.parallel.moe_parallel.moe_pipeline.layers.moe_layers import MoeSparseBlockDPModule, MoeSparseBlockPipeModule
from mytransformers.utils import Logger

class MoeSparseBlockDPModuleGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: Module,
                gate: Callable,
                main_rank: int,
                buffer: Tensor,
                replace_experts_layer: MoeExperts,
                expert_idxs: List[Tensor],
                moe_group: ProcessGroup,
                device: torch.device) -> MoeExperts:
        experts = MoeExpertsModuleGenerator(module.experts,
                                            replace_experts_layer,
                                            expert_idxs,
                                            moe_group,
                                            device)
        return MoeSparseBlockDPModule(experts, gate, moe_group, main_rank, buffer).to(device)
    
class MoeSparseBlockPipeModuleGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: Module,
                gate: Callable,
                main_rank: int,
                replace_experts_layer: MoeExperts,
                expert_idxs: List[Tensor],
                moe_group: ProcessGroup,
                device: torch.device) -> MoeExperts:
        experts = MoeExpertsModuleGenerator(module.experts,
                                            replace_experts_layer,
                                            expert_idxs,
                                            moe_group,
                                            device,
                                            main_rank=main_rank)
        return MoeSparseBlockPipeModule(experts, gate, moe_group, main_rank).to(device)
        
        
    