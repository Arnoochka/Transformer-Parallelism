import torch
from torch.distributed import ProcessGroup
import torch.distributed as dist
from torch import Tensor
from .MoeExperts import MoeExperts
from torch.nn import ModuleList
from enum import Enum

class LayerRole(Enum):
    main = "main"
    assistant = "assistant"

class MoePipeExpertsSpeed(MoeExperts):
    """
    Реализация MoE параллелизма с Pipeline параллелизмом, ориентированная на пропускную способность.
    
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
                 moe_group: ProcessGroup,
                 is_main: LayerRole):
        super().__init__(global_num_experts,
                         local_experts,
                         expert_to_rank,
                         global_to_local_expert_idxs,
                         moe_group)
        self.is_main = is_main
        
    @torch.no_grad()
    def forward(self,
                hidden_states: Tensor,
                top_k_index: Tensor,
                top_k_weights: Tensor) -> Tensor:
        """
        input:
            X (main) | Fake (assistant)
            X.size = (batch_size * seq_len, hidden_dim)
            (top_k.size = (batch_size * seq_len, k))
        Output:
            X (main) | Fake (assistant)
        """
        
        if self.is_main == LayerRole.assistant:
            scatter_list = None
        else:
            pass
            
        
        
        
