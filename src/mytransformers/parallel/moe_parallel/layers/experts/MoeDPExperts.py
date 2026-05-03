import torch
import torch.distributed as dist
from torch import Tensor
from .MoeExperts import MoeExperts


class MoeDPExperts(MoeExperts):

    @torch.no_grad()
    def forward(self,
                hidden_states: Tensor,
                full_top_k_index: Tensor,
                top_k_weights: Tensor) -> Tensor:
        num_tokens, hidden_dim = hidden_states.size()
        k = full_top_k_index.size(1)

        flat_topk = full_top_k_index.reshape(-1)
        global_ranks = self.expert_to_rank[flat_topk]

        my_start = self.rank * num_tokens * k
        my_end = my_start + num_tokens * k
        local_ranks = global_ranks[my_start:my_end]

        local_sort_perm = torch.argsort(local_ranks, stable=True) 
        send_counts = torch.bincount(local_ranks, minlength=self.world_size)

        global_ranks_2d = global_ranks.view(self.world_size, num_tokens * k)
        recv_counts = (global_ranks_2d == self.rank).sum(dim=1)
        del global_ranks_2d

        global_sort_perm = torch.argsort(global_ranks, stable=True)
        recv_mask = global_ranks[global_sort_perm] == self.rank
        del global_ranks

        recv_expert_ids = flat_topk[global_sort_perm[recv_mask]]
        del global_sort_perm, recv_mask, flat_topk

        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()
        total_recv = recv_counts.sum().item()

        token_ids = local_sort_perm // k
        send_buf = hidden_states[token_ids]
        del token_ids

        recv_hidden = torch.empty(
            (total_recv, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device)

        self.scheduler.transfer(
            op=dist.all_to_all_single,
            op_info=self.thread_idx,
            op_name="first_all_to_all",
            output=recv_hidden,
            input=send_buf,
            output_split_sizes=recv_counts_list,
            input_split_sizes=send_counts_list,
            group=self.moe_group)
        del send_buf

        local_expert_ids = self.global_to_local_expert_idxs[recv_expert_ids]
        del recv_expert_ids

        expert_sort_perm = torch.argsort(local_expert_ids, stable=True)

        num_local = len(self.local_experts)
        counts = torch.bincount(local_expert_ids, minlength=num_local)
        del local_expert_ids
        offsets = torch.zeros(num_local + 1, dtype=torch.long,
                              device=hidden_states.device)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        del counts

        recv_hidden = recv_hidden[expert_sort_perm]

        recv_hidden = self.compute(recv_hidden, offsets)
        del offsets

        expert_unsort = torch.empty_like(expert_sort_perm)
        expert_unsort[expert_sort_perm] = torch.arange(
            total_recv, device=hidden_states.device)
        del expert_sort_perm

        recv_hidden = recv_hidden[expert_unsort]
        del expert_unsort

        result = torch.empty(
            (num_tokens * k, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device)

        self.scheduler.transfer(
            op=dist.all_to_all_single,
            op_info=self.thread_idx,
            op_name="second_all_to_all",
            output=result,
            input=recv_hidden,
            output_split_sizes=send_counts_list,
            input_split_sizes=recv_counts_list,
            group=self.moe_group)
        del recv_hidden

        token_unsort = torch.empty_like(local_sort_perm)
        token_unsort[local_sort_perm] = torch.arange(
            local_sort_perm.size(0), device=hidden_states.device)
        del local_sort_perm

        result = result[token_unsort]
        del token_unsort

        result.mul_(top_k_weights.reshape(-1).unsqueeze(-1))
        output = result.view(num_tokens, k, hidden_dim).sum(dim=1)
        del result

        return output