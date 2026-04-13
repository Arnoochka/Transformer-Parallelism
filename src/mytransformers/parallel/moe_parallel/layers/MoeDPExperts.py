import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional
from .MoeExperts import MoeExperts


class MoeDPExpertsMemory(MoeExperts):
    @torch.no_grad()
    def forward(self,
                hidden_states: Tensor,
                top_k_index: Tensor,
                top_k_weights: Tensor) -> Tensor:

        num_tokens, k = top_k_index.size()
        hidden_dim = hidden_states.size(1)

        full_top_k_index = torch.empty(
            (num_tokens * self.world_size, k),
            dtype=top_k_index.dtype,
            device=top_k_index.device
        )

        dist.all_gather_into_tensor(full_top_k_index, top_k_index, group=self.moe_group)

        final_hidden_states = torch.zeros_like(hidden_states)

        for i in range(k):
            global_experts = full_top_k_index[:, i]
            global_ranks = self.expert_to_rank[global_experts]

            local_slice = slice(num_tokens * self.rank, num_tokens * (self.rank + 1))
            local_ranks = global_ranks[local_slice]

            global_sorted_ranks, global_sorted_indices = torch.sort(global_ranks, stable=True)
            local_sorted_ranks, local_sorted_indices = torch.sort(local_ranks, stable=True)

            send_counts = torch.bincount(local_sorted_ranks, minlength=self.world_size)

            global_ranks_2d = global_ranks.view(self.world_size, num_tokens)
            recv_counts = (global_ranks_2d == self.rank).sum(dim=1)

            send_counts_list = send_counts.tolist()
            recv_counts_list = recv_counts.tolist()

            need_mask = (global_sorted_ranks == self.rank)
            need_indices = global_experts[global_sorted_indices][need_mask]

            need_hidden_states = torch.empty(
                (recv_counts.sum().item(), hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

            expert_mask = functional.one_hot(
                self.global_to_local_expert_idxs[need_indices],
                num_classes=len(self.local_experts)
            ).transpose(0, 1).to(torch.bool)

            dist.all_to_all_single(
                need_hidden_states,
                hidden_states[local_sorted_indices],
                recv_counts_list,
                send_counts_list,
                group=self.moe_group
            )

            need_hidden_states = self.compute(need_hidden_states, expert_mask)

            output = torch.empty_like(hidden_states)

            dist.all_to_all_single(
                output,
                need_hidden_states,
                send_counts_list,
                recv_counts_list,
                group=self.moe_group
            )

            output.mul_(top_k_weights[:, i][local_sorted_indices].unsqueeze(-1))

            final_hidden_states.index_add_(0, local_sorted_indices, output)

        return final_hidden_states
    
    
class MoeDPExpertsSpeed(MoeExperts):

    @torch.no_grad()
    def forward(self,
                hidden_states: Tensor,
                top_k_index: Tensor,
                top_k_weights: Tensor) -> Tensor:

        num_tokens, k = top_k_index.size()
        hidden_dim = hidden_states.size(1)

        full_top_k_index = torch.empty(
            (num_tokens * self.world_size, k),
            dtype=top_k_index.dtype,
            device=top_k_index.device
        )

        dist.all_gather_into_tensor(full_top_k_index, top_k_index, group=self.moe_group)

        flat_topk = full_top_k_index.reshape(-1)
        global_ranks = self.expert_to_rank[flat_topk]

        token_indices = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(k)

        local_slice = slice(num_tokens * self.rank * k, num_tokens * (self.rank + 1) * k)
        local_ranks = global_ranks[local_slice]

        global_sorted_ranks, global_sorted_indices = torch.sort(global_ranks, stable=True)
        local_sorted_ranks, local_sorted_indices = torch.sort(local_ranks, stable=True)

        send_counts = torch.bincount(local_sorted_ranks, minlength=self.world_size)

        global_ranks_2d = global_ranks.view(self.world_size, num_tokens * k)
        recv_counts = (global_ranks_2d == self.rank).sum(dim=1)

        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()

        need_mask = (global_sorted_ranks == self.rank)
        need_indices = flat_topk[global_sorted_indices][need_mask]
        
        need_hidden_states = torch.empty(
            (recv_counts.sum().item(), hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        expert_mask = functional.one_hot(
            self.global_to_local_expert_idxs[need_indices],
            num_classes=len(self.local_experts)
        ).transpose(0, 1).to(torch.bool)

        sorted_token_indices = token_indices[local_sorted_indices]

        dist.all_to_all_single(
            need_hidden_states,
            hidden_states[sorted_token_indices],
            recv_counts_list,
            send_counts_list,
            group=self.moe_group
        )

        need_hidden_states = self.compute(need_hidden_states, expert_mask)

        computed = torch.empty(
            (num_tokens * k, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        dist.all_to_all_single(
            computed,
            need_hidden_states,
            send_counts_list,
            recv_counts_list,
            group=self.moe_group
        )

        inverse_indices = torch.empty_like(local_sorted_indices)
        inverse_indices[local_sorted_indices] = torch.arange(
            local_sorted_indices.size(0),
            device=local_sorted_indices.device
        )

        computed = computed[inverse_indices]
        flat_weights = top_k_weights.reshape(-1)
        computed.mul_(flat_weights.unsqueeze(-1))

        return computed.view(num_tokens, k, hidden_dim).sum(dim=1)
