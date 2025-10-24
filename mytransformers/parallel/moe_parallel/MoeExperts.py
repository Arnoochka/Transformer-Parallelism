import torch
from torch.distributed import ProcessGroup
import torch.distributed as dist
from torch import Tensor
from torch.nn import ModuleList, functional
from mytransformers.parallel.ParallelModule import ParallelModule

class MoeExperts(ParallelModule):
    def __init__(self,
                 num_experts: int,
                 moe_group: ProcessGroup):
        super().__init__()
        
        self.num_experts = num_experts
        self.moe_group = moe_group
        
    def compute(self,
                hidden_states: Tensor,
                experts: ModuleList,
                expert_mask: Tensor
                ) -> Tensor:
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idxs = torch.where(expert_mask[expert_idx].squeeze(0))
            hidden_states[idxs] = experts[expert_idx](hidden_states[idxs])
            
        return hidden_states
    
class MoeDPExperts(MoeExperts):
    def __init__(self,
                 num_experts: int,
                 local_experts: ModuleList,
                 local_expert_idxs: Tensor,
                 expert_to_rank: Tensor,
                 moe_group: ProcessGroup):
        super().__init__(num_experts, moe_group)

        self.local_experts = local_experts
        self.rank = dist.get_rank(group=moe_group)
        self.world_size = dist.get_world_size(group=moe_group)
        self.local_expert_idxs = local_expert_idxs
        self.expert_to_rank = expert_to_rank
           
    @torch.no_grad()
    def forward(self,
                hidden_states: Tensor,
                top_k_index: Tensor,
                top_k_weights: Tensor) -> torch.Tensor:
        
        """
        Data: X = [X_1, X_2] (dim=0)
        Data: (batch_size * seq_len, hidden_dim (or k))
        """
        num_tokens, k = top_k_index.size()
        hidden_dim = hidden_states.size(1)
        full_top_k_index = torch.empty(
            size=(num_tokens * self.world_size, k),
            layout=top_k_index.layout,
            dtype=top_k_index.dtype,
            device=top_k_index.device) 
        
        dist.all_gather_into_tensor(full_top_k_index, top_k_index, group=self.moe_group)
        final_hidden_states = torch.zeros_like(hidden_states)
        for i in range(k):
            global_ranks = self.expert_to_rank[full_top_k_index[:, i]]
            local_ranks = global_ranks[num_tokens * self.rank: num_tokens * (self.rank + 1)]
            local_sorted_ranks, local_sorted_indices = torch.sort(local_ranks, stable=True)
            send_counts = torch.bincount(local_sorted_ranks, minlength=self.world_size)
            recv_counts = torch.empty_like(send_counts)
            
            for rank in range(self.world_size):
                device_ranks = global_ranks[num_tokens * self.rank: num_tokens * (self.rank + 1)]
                recv_counts[rank] = (device_ranks == self.rank).sum().item()
                
            need_hidden_states = torch.empty(
                size=(recv_counts.sum().item(), hidden_dim),
                layout=hidden_states.layout,
                device=hidden_states.device,
                dtype=hidden_states.dtype)
            
            need_indices = torch.empty(
                size=(recv_counts.sum().item(),),
                layout=top_k_index.layout,
                device=top_k_index.device,
                dtype=top_k_index.dtype)
            
            recv_cumsum = torch.cumsum(recv_counts)
            recv_global_indices = full_top_k_index[global_ranks == self.rank, i]
            start = 0
            for rank in range(self.world_size):
                end = recv_cumsum[rank]
                if start == end: continue
                recv_rank_indices = recv_global_indices[start:end]
                recv_rank_sorted_indices, _ = torch.sort(
                    recv_rank_indices, stable=True)
                need_indices[start:end] = recv_rank_sorted_indices
                start = end
                
            expert_mask = functional.one_hot(need_indices, num_classes=len(self.local_experts)).T
            dist.all_to_all_single(
                need_hidden_states,
                hidden_states[local_sorted_indices],
                recv_counts, send_counts,
                group=self.moe_group)

            need_hidden_states = self.compute(need_hidden_states, self.local_experts, expert_mask)
            computed_hidden_states = torch.empty_like(hidden_states[local_sorted_indices])
            dist.all_to_all_single(
                computed_hidden_states,
                need_hidden_states,
                send_counts, recv_counts,
                group=self.moe_group)
            
            final_hidden_states[local_sorted_indices] += top_k_weights[:, i] * computed_hidden_states
            
        return final_hidden_states