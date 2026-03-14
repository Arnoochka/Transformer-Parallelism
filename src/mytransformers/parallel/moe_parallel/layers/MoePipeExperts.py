import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import ModuleList, functional
from .MoeExperts import MoeExperts

class MoePipeExpertsMemory(MoeExperts):

    def __init__(self,
                 global_num_experts: int,
                 local_experts: ModuleList,
                 expert_to_rank: Tensor,
                 global_to_local_expert_idxs: Tensor,
                 moe_group,
                 main_rank: int):
        super().__init__(global_num_experts,
                         local_experts,
                         expert_to_rank,
                         global_to_local_expert_idxs,
                         moe_group)

        self.main_rank = main_rank

    @torch.no_grad()
    def forward(self,
                hidden_states: Tensor,
                top_k_index: Tensor,
                top_k_weights: Tensor) -> Tensor:

        _, k = top_k_index.size()
        hidden_dim = hidden_states.size(1)
        if self.rank == self.main_rank:
            final_hidden_states = torch.zeros_like(hidden_states)

        for i in range(k):
            recv_count = torch.empty((1,),
                                     dtype=torch.long,
                                     device=hidden_states.device)
            if self.rank == self.main_rank:
                global_ranks = self.expert_to_rank[top_k_index[:, i]]
                global_sorted_ranks, global_sorted_indices = torch.sort(global_ranks, stable=True)
                send_counts = torch.bincount(global_sorted_ranks, minlength=self.world_size)

                count_split = [send_counts[r].view(1) for r in range(self.world_size)]
                hidden_split = list(torch.split(hidden_states[global_sorted_indices], send_counts.tolist()))
                index_spit = list(torch.split(top_k_index[global_sorted_indices, i], send_counts.tolist()))
                weight_split = list(torch.split(top_k_weights[global_sorted_indices, i], send_counts.tolist()))
                
                gather_hidden = [torch.empty_like(hidden_split[r]) for r in range(self.world_size)]
            else:
                count_split = None
                hidden_split = None
                index_spit = None
                weight_split = None
                gather_hidden = None

            dist.scatter(recv_count, scatter_list=count_split, src=self.main_rank, group=self.moe_group)
            
            local_count = recv_count.item()
            local_hidden_states = torch.empty((local_count, hidden_dim),
                                              dtype=hidden_states.dtype,
                                              device=hidden_states.device)
            local_k_weights = torch.empty((local_count,),
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)
            local_k_index = torch.empty((local_count,),
                                        dtype=hidden_states.dtype,
                                        device=hidden_states.device)
            
            dist.scatter(local_hidden_states, scatter_list=hidden_split, src=self.main_rank, group=self.moe_group)
            dist.scatter(local_k_index, scatter_list=index_spit, src=self.main_rank, group=self.moe_group)
            dist.scatter(local_k_weights, scatter_list=weight_split, src=self.main_rank, group=self.moe_group)
            
            local_k_index = self.global_to_local_expert_idxs[local_k_index]
            expert_mask = functional.one_hot(
                self.global_to_local_expert_idxs[local_k_index],
                num_classes=len(self.local_experts)
            ).transpose(0, 1).to(torch.bool)
            
            local_hidden_states = self.compute(local_hidden_states, expert_mask)
            local_hidden_states.mul_(local_k_weights.unsqueeze(-1))
            dist.gather(local_hidden_states, gather_list=gather_hidden, dst=self.main_rank, group=self.moe_group)
            final_hidden_states
            
        if self.rank == self.main_rank:
            gathered = torch.cat(gather_hidden, dim=0)
            final_hidden_states.index_add_(0, global_sorted_indices, gathered)
            
        if self.rank == self.main_rank:
            return final_hidden_states
        else: 
            return hidden_states

class MoePipeExpertsSpeed(MoeExperts):
    def __init__(self,
                 global_num_experts: int,
                 local_experts: ModuleList,
                 expert_to_rank: Tensor,
                 global_to_local_expert_idxs: Tensor,
                 moe_group,
                 main_rank: int):
        super().__init__(global_num_experts,
                         local_experts,
                         expert_to_rank,
                         global_to_local_expert_idxs,
                         moe_group)
        
        self.main_rank = main_rank

    @torch.no_grad()
    def forward(self,
                hidden_states: Tensor,
                top_k_index: Tensor,
                top_k_weights: Tensor) -> Tensor:

        num_tokens, k = top_k_index.size()
        hidden_dim = hidden_states.size(1)

        expanded_hidden = hidden_states.repeat_interleave(k, dim=0)
        flat_index = top_k_index.reshape(-1)
        flat_weights = top_k_weights.reshape(-1)

        recv_count = torch.empty((1,), dtype=torch.long, device=hidden_states.device)

        if self.rank == self.main_rank:
            global_ranks = self.expert_to_rank[flat_index]
            global_sorted_ranks, global_sorted_indices = torch.sort(global_ranks, stable=True)
            send_counts = torch.bincount(global_sorted_ranks, minlength=self.world_size)
            count_split = [send_counts[r].view(1) for r in range(self.world_size)]

            hidden_split = list(torch.split(expanded_hidden[global_sorted_indices], send_counts.tolist()))
            index_split = list(torch.split(flat_index[global_sorted_indices], send_counts.tolist()))
            weight_split = list(torch.split(flat_weights[global_sorted_indices], send_counts.tolist()))

            gather_hidden = [torch.empty_like(hidden_split[r]) for r in range(self.world_size)]

        else:
            count_split = None
            hidden_split = None
            index_split = None
            weight_split = None
            gather_hidden = None

        dist.scatter(recv_count, scatter_list=count_split, src=self.main_rank, group=self.moe_group)

        local_count = recv_count.item()
        local_hidden_states = torch.empty((local_count, hidden_dim),
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)
        local_k_weights = torch.empty((local_count,),
                                      dtype=hidden_states.dtype,
                                      device=hidden_states.device)
        local_k_index = torch.empty((local_count,),
                                    dtype=torch.long,
                                    device=hidden_states.device)

        dist.scatter(local_hidden_states, scatter_list=hidden_split, src=self.main_rank, group=self.moe_group)
        dist.scatter(local_k_index, scatter_list=index_split, src=self.main_rank, group=self.moe_group)
        dist.scatter(local_k_weights, scatter_list=weight_split, src=self.main_rank, group=self.moe_group)

        local_k_index = self.global_to_local_expert_idxs[local_k_index]

        expert_mask = functional.one_hot(
            local_k_index,
            num_classes=len(self.local_experts)
        ).transpose(0, 1).to(torch.bool)

        local_hidden_states = self.compute(local_hidden_states, expert_mask)

        local_hidden_states.mul_(local_k_weights.unsqueeze(-1))

        dist.gather(local_hidden_states,
                    gather_list=gather_hidden,
                    dst=self.main_rank,
                    group=self.moe_group)

        if self.rank == self.main_rank:
            gathered = torch.cat(gather_hidden, dim=0)
            inverse_indices = torch.empty_like(global_sorted_indices)
            inverse_indices[global_sorted_indices] = torch.arange(global_sorted_indices.size(0),
                                                                  device=global_sorted_indices.device)
            gathered = gathered[inverse_indices]
            gathered = gathered.view(num_tokens, k, hidden_dim)
            
            return gathered.sum(dim=1)
        else:
            return torch.empty_like(hidden_states)