import torch
from torch.distributed import ProcessGroup
from typing import List
import torch.distributed as dist
from torch import Tensor
from torch.nn import ModuleList
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from .MoeExperts import MoeDPExperts

class MoeDPExpertsGenerator(ParallelModuleGenerator):
    """
    генерирует MoeDPExperts из модуля экспертов модели
    
    Args:
        module (List[ModuleList]): исходный MoE слой
        expert_idxs (List[Tensor]): план распределения экспертов по устройствам
        moe_group: (ProcessGroup): Группа процессов MoE
        device: устройство на котором будет работать эксперт
    """
    def __new__(cls,
                module: ModuleList,
                expert_idxs: List[Tensor],
                moe_group: ProcessGroup,
                device: torch.device) -> MoeDPExperts:
        rank = dist.get_rank(moe_group)
        world_size = dist.get_world_size(moe_group)
        expert_ranks = expert_idxs[rank]
        num_experts = module.num_experts
        expert_to_rank = torch.empty(num_experts, dtype=torch.int64)
        for rank, expert_idxs in enumerate(expert_idxs):
            expert_to_rank[expert_idxs] = rank
        expert_to_rank = expert_to_rank.to(device)
        global_to_local_expert_idxs = torch.empty_like(expert_to_rank)
        local_expert_idxs = torch.arange(
            expert_ranks.size(0),
            dtype=expert_to_rank.dtype,
            device=expert_to_rank.device)      
        global_to_local_expert_idxs = [torch.empty_like(local_expert_idxs)
                                       for _ in range(world_size)]
        dist.all_gather(global_to_local_expert_idxs, local_expert_idxs, group=moe_group)
        global_to_local_expert_idxs = torch.cat(global_to_local_expert_idxs, dim=0)
            
        local_experts = ModuleList([module[r.item()] for r in expert_ranks])
        global_expert_idxs = expert_idxs[rank].to(device)
        
        return MoeDPExperts(num_experts,
                            local_experts,
                            global_expert_idxs,
                            expert_to_rank,
                            global_to_local_expert_idxs,
                            moe_group)
        