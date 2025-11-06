import torch
from torch.distributed import ProcessGroup
import torch.distributed as dist
from torch import Tensor
from torch.nn import ModuleList, functional
from mytransformers.parallel.ParallelModule import ParallelModule
from mytransformers.benchmark import Tracker
from mytransformers.utils import Logger

def synchronize():
        dist.barrier(None)

class MoeExperts(ParallelModule):
    def __init__(self,
                 global_num_experts: int,
                 experts: ModuleList,
                 moe_group: ProcessGroup):
        super().__init__()
        self.tracker = Tracker(None, synchronize)
        self.tracker.start()
        self.global_num_experts = global_num_experts
        self.experts = experts
        self.moe_group = moe_group
        
    def compute(self,
                hidden_states: Tensor,
                expert_mask: Tensor
                ) -> Tensor:
        expert_hit = torch.greater(expert_mask.sum(dim=-1), 0).nonzero()
        for expert_idx in expert_hit:
            idxs = torch.where(expert_mask[expert_idx].squeeze(0))
            hidden_states[idxs] = self.experts[expert_idx](hidden_states[idxs])
            
        return hidden_states
    
class MoeDPExperts(MoeExperts):
    def __init__(self,
                 global_num_experts: int,
                 local_experts: ModuleList,
                 global_expert_idxs: Tensor,
                 expert_to_rank: Tensor,
                 global_to_local_expert_idxs: Tensor,
                 moe_group: ProcessGroup):
        super().__init__(global_num_experts, local_experts, moe_group)
        self.rank = dist.get_rank(group=moe_group)
        self.world_size = dist.get_world_size(group=moe_group)
        self.global_expert_idxs = global_expert_idxs
        self.expert_to_rank = expert_to_rank
        self.global_to_local_expert_idxs = global_to_local_expert_idxs
    @torch.no_grad()
    def forward(self,
                hidden_states: Tensor,
                top_k_index: Tensor,
                top_k_weights: Tensor) -> Tensor:
        
        """
        Data: X = [X_1, X_2] (dim=0)
        Data: (batch_size * seq_len, hidden_dim (or k))
        """
        self.tracker.snapshot("start forward")
        num_tokens, k = top_k_index.size()
        hidden_dim = hidden_states.size(1)
        full_top_k_index = torch.empty(
            size=(num_tokens * self.world_size, k),
            layout=top_k_index.layout,
            dtype=top_k_index.dtype,
            device=top_k_index.device) 
        
        dist.all_gather_into_tensor(full_top_k_index, top_k_index, group=self.moe_group)
        self.tracker.snapshot("full top k index")
        final_hidden_states = torch.zeros_like(hidden_states)
        self.tracker.snapshot("final hidden states")
        for i in range(k):
            global_ranks = self.expert_to_rank[full_top_k_index[:, i]]
            local_ranks = global_ranks[num_tokens * self.rank: num_tokens * (self.rank + 1)]
            global_sorted_ranks, global_sorted_indices = torch.sort(global_ranks, stable=True)   
            self.tracker.snapshot("sort global ranks")
            need_indices = full_top_k_index[global_sorted_indices, i][global_sorted_ranks == self.rank]
            self.tracker.snapshot("need indicies")
            local_sorted_ranks, local_sorted_indices = torch.sort(local_ranks, stable=True)
            self.tracker.snapshot("local sorted ranks")
            send_counts = torch.bincount(local_sorted_ranks, minlength=self.world_size)
            self.tracker.snapshot("bincount")
            recv_counts = torch.empty_like(send_counts)
            self.tracker.snapshot("recv counts")
            for rank in range(self.world_size):
                device_ranks = global_ranks[num_tokens * rank: num_tokens * (rank + 1)]
                recv_counts[rank] = (device_ranks == self.rank).sum().item()
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            self.tracker.snapshot("counts.tolist")
            need_hidden_states = torch.empty(
                size=(sum(recv_counts), hidden_dim),
                layout=hidden_states.layout,
                device=hidden_states.device,
                dtype=hidden_states.dtype)
            self.tracker.snapshot("need hidden states")
            expert_mask = functional.one_hot(
                self.global_to_local_expert_idxs[need_indices],
                num_classes=len(self.experts)
            ).transpose(0, 1)
            self.tracker.snapshot("expert mask")
            dist.all_to_all_single(
                need_hidden_states,
                hidden_states[local_sorted_indices],
                recv_counts, send_counts,
                group=self.moe_group)
            self.tracker.snapshot("first all2all")
            need_hidden_states = self.compute(need_hidden_states, expert_mask)
            self.tracker.snapshot("compute")
            computed_hidden_states = torch.empty_like(hidden_states)
            self.tracker.snapshot("computed hidden states")
            dist.all_to_all_single(
                computed_hidden_states,
                need_hidden_states,
                send_counts, recv_counts,
                group=self.moe_group)
            self.tracker.snapshot("second all2all")
            computed_hidden_states.mul_(top_k_weights[:, i][local_sorted_indices].unsqueeze(-1))
            self.tracker.snapshot("final mul")
            final_hidden_states.index_add_(0, local_sorted_indices, computed_hidden_states)
            self.tracker.snapshot("final add")
        self.tracker.snapshot("return")
        return final_hidden_states