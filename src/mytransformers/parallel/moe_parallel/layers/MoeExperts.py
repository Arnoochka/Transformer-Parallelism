import torch
from torch.distributed import ProcessGroup
from torch import Tensor
from torch.nn import ModuleList
from mytransformers.parallel.ParallelModule import ParallelModule
import torch.distributed as dist


class MoeExperts(ParallelModule):
    """
    Базовый вычислительный класс Mixture of Experts параллелизма.

    Args:
        global_num_experts (int): Общее количество экспертов.
        local_experts (ModuleList): Локальный список экспертов.
        expert_to_rank (Tensor): Отображение эксперта на ранг устройства.
        global_to_local_expert_idxs (Tensor): Сопоставление глобальных индексов экспертов с локальными.
        moe_group (ProcessGroup): Группа процессов MoE.
    """
    def __init__(self,
                 global_num_experts: int,
                 local_experts: ModuleList,
                 expert_to_rank: Tensor,
                 global_to_local_expert_idxs: Tensor,
                 moe_group: ProcessGroup):
        super().__init__()
        self.rank = dist.get_rank(group=moe_group)
        self.world_size = dist.get_world_size(group=moe_group)
        
        self.global_num_experts = global_num_experts
        self.local_experts = local_experts
        self.expert_to_rank = expert_to_rank
        self.global_to_local_expert_idxs = global_to_local_expert_idxs
        self.moe_group = moe_group

    def compute(self, hidden_states: Tensor, expert_mask: Tensor) -> Tensor:
        """
        hidden_states: (num_tokens, hidden_dim)
        expert_mask: (num_experts_local, num_tokens)
        """
    
        for expert_idx, expert in enumerate(self.local_experts):
            idxs = torch.nonzero(expert_mask[expert_idx], as_tuple=True)[0]
    
            if idxs.numel() == 0:
                continue
            
            expert_input = hidden_states.index_select(0, idxs)
            expert_output = expert(expert_input)
    
            hidden_states.index_copy_(0, idxs, expert_output)
    
        return hidden_states