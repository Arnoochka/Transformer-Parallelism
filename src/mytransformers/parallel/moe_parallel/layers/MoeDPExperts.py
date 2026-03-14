import torch
from torch.distributed import ProcessGroup
import torch.distributed as dist
from torch import Tensor
from torch.nn import ModuleList, functional
from mytransformers.parallel.ParallelModule import ParallelModule
from .MoeExperts import MoeExperts


class MoeDPExpertsMemory(MoeExperts):
    """
    Реализация MoE параллелизма с параллелизмом по данным, ориентированная на экономию памяти.
    """
    @torch.no_grad()
    def forward(self,
                hidden_states: Tensor,
                top_k_index: Tensor,
                top_k_weights: Tensor) -> Tensor:
        """
        input:
            X = [X_1, X_2] (dim=0)
            X.size = (batch_size * seq_len, hidden_dim)
            (top_k.size = (batch_size * seq_len, k))
        Output:
            X = [X_1, X_2] (dim 0)
        """
        num_tokens, k = top_k_index.size()
        hidden_dim = hidden_states.size(1)

        full_top_k_index = torch.empty(
            size=(num_tokens * self.world_size, k),
            layout=top_k_index.layout,
            dtype=top_k_index.dtype,
            device=top_k_index.device
        )

        dist.all_gather_into_tensor(full_top_k_index, top_k_index, group=self.moe_group)
        final_hidden_states = torch.zeros_like(hidden_states)

        for i in range(k):
            global_ranks = self.expert_to_rank[full_top_k_index[:, i]]
            local_ranks = global_ranks[num_tokens * self.rank: num_tokens * (self.rank + 1)]
            global_sorted_ranks, global_sorted_indices = torch.sort(global_ranks, stable=True)
            need_indices = full_top_k_index[global_sorted_indices, i][global_sorted_ranks == self.rank]
            local_sorted_ranks, local_sorted_indices = torch.sort(local_ranks, stable=True)

            send_counts = torch.bincount(local_sorted_ranks, minlength=self.world_size)
            recv_counts = torch.empty_like(send_counts)
            for rank in range(self.world_size):
                device_ranks = global_ranks[num_tokens * rank: num_tokens * (rank + 1)]
                recv_counts[rank] = (device_ranks == self.rank).sum().item()

            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()

            need_hidden_states = torch.empty(
                size=(sum(recv_counts), hidden_dim),
                layout=hidden_states.layout,
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )

            expert_mask = functional.one_hot(
                self.global_to_local_expert_idxs[need_indices],
                num_classes=len(self.local_experts)
            ).transpose(0, 1).to(torch.bool)

            dist.all_to_all_single(
                need_hidden_states,
                hidden_states[local_sorted_indices],
                recv_counts, send_counts,
                group=self.moe_group
            )

            need_hidden_states = self.compute(need_hidden_states, expert_mask)

            computed_hidden_states = torch.empty_like(hidden_states)
            dist.all_to_all_single(
                computed_hidden_states,
                need_hidden_states,
                send_counts, recv_counts,
                group=self.moe_group
            )

            computed_hidden_states.mul_(top_k_weights[:, i][local_sorted_indices].unsqueeze(-1))
            final_hidden_states.index_add_(0, local_sorted_indices, computed_hidden_states)
        return final_hidden_states
    
    
class MoeDPExpertsSpeed(MoeExperts):
    """
    Реализация MoE параллелизма с параллелизмом по данным, ориентированная на пропускную способность.
    """
    @torch.no_grad()
    def forward(self,
                hidden_states: Tensor,
                top_k_index: Tensor,
                top_k_weights: Tensor) -> Tensor:
        """
        input:
            X = [X_1, X_2] (dim=0)
            X.size = (batch_size * seq_len, hidden_dim)
            top_k_index.size = (num_tokens, k)
        """
        num_tokens, k = top_k_index.size()
        hidden_dim = hidden_states.size(1)

        full_top_k_index = torch.empty(
            size=(num_tokens * self.world_size, k),
            layout=top_k_index.layout,
            dtype=top_k_index.dtype,
            device=top_k_index.device
        )

        dist.all_gather_into_tensor(full_top_k_index, top_k_index, group=self.moe_group)

        flat_topk = full_top_k_index.reshape(-1)
        global_ranks = self.expert_to_rank[flat_topk]
        local_ranks = global_ranks[num_tokens * self.rank * k: num_tokens * (self.rank + 1) * k]
        global_sorted_ranks, global_sorted_indices = torch.sort(global_ranks, stable=True)
        need_indices = flat_topk[global_sorted_indices][global_sorted_ranks == self.rank]
        local_sorted_ranks, local_sorted_indices = torch.sort(local_ranks, stable=True)

        send_counts = torch.bincount(local_sorted_ranks, minlength=self.world_size)
        recv_counts = torch.empty_like(send_counts)
        
        for rank in range(self.world_size):
            device_ranks = global_ranks[rank * num_tokens * k: (rank + 1) * num_tokens * k]
            recv_counts[rank] = (device_ranks == self.rank).sum().item()

        send_counts = send_counts.tolist()
        recv_counts = recv_counts.tolist()

        need_hidden_states = torch.empty(
            size=(sum(recv_counts), hidden_dim),
            layout=hidden_states.layout,
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )

        expert_mask = functional.one_hot(
            self.global_to_local_expert_idxs[need_indices],
            num_classes=len(self.local_experts)
        ).transpose(0, 1).to(torch.bool)

        expanded_hidden = hidden_states.repeat_interleave(k, dim=0)

        dist.all_to_all_single(
            need_hidden_states,
            expanded_hidden[local_sorted_indices],
            recv_counts, send_counts,
            group=self.moe_group
        )
        
        need_hidden_states = self.compute(need_hidden_states, expert_mask)

        computed_hidden_states = torch.empty_like(expanded_hidden)

        dist.all_to_all_single(
            computed_hidden_states,
            need_hidden_states,
            send_counts, recv_counts,
            group=self.moe_group
        )

        inverse_indices = torch.empty_like(local_sorted_indices)
        inverse_indices[local_sorted_indices] = torch.arange(
            local_sorted_indices.size(0),
            device=local_sorted_indices.device
        )
        
        computed_hidden_states = computed_hidden_states[inverse_indices]
        
        flat_weights = top_k_weights.reshape(-1)
        computed_hidden_states.mul_(flat_weights.unsqueeze(-1))
        
        hidden_states = computed_hidden_states.view(num_tokens, k, hidden_dim).sum(dim=1)
        return hidden_states
