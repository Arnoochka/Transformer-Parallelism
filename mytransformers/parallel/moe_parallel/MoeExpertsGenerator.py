import torch
from torch.distributed import ProcessGroup
from typing import List
import torch.distributed as dist
from torch import Tensor
from torch.nn import ModuleList
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from .MoeExperts import MoeDPExperts

class MoeDataParallelExpertsGenerator(ParallelModuleGenerator):
    expert_idxs: List[Tensor] = None
    moe_group: ProcessGroup = None
    def __new__(cls, module: ModuleList, device: torch.device) -> MoeDPExperts:
        rank = dist.get_rank()
        expert_ranks = cls.expert_idxs[rank]
        local_experts = ModuleList([module[r] for r in expert_ranks]).to(device)
        num_experts = module.num_experts
        local_expert_idxs = cls.expert_idxs[rank].to(device)
        expert_to_rank = torch.empty(num_experts, dtype=torch.int64)
        for rank, expert_idxs in enumerate(cls.expert_idxs):
            expert_to_rank[expert_idxs] = rank
        expert_to_rank = expert_to_rank.to(device)
        
        return MoeDPExperts(num_experts,
                            local_experts,
                            local_expert_idxs,
                            expert_to_rank,
                            cls.moe_group)
        