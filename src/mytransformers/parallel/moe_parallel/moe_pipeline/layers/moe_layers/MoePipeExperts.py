import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import ModuleList
from .MoeExperts import MoeExperts
from torch.distributed import ProcessGroup

class MoePipeExpertsMemory(MoeExperts):

    def __init__(self,
                 global_num_experts: int,
                 local_experts: ModuleList,
                 expert_to_rank: Tensor,
                 global_to_local_expert_idxs: Tensor,
                 moe_group: ProcessGroup,
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
            recv_info = torch.empty((2,),
                                    dtype=torch.long,
                                    device=hidden_states.device)
            if self.rank == self.main_rank:
                global_ranks = self.expert_to_rank[top_k_index[:, i]]
                global_sorted_ranks, global_sorted_indices = torch.sort(global_ranks, stable=True)
                sorted_hidden = hidden_states[global_sorted_indices]
                sorted_index = top_k_index[global_sorted_indices, i]
                sorted_weights = top_k_weights[global_sorted_indices, i]
                send_counts = torch.bincount(global_sorted_ranks, minlength=self.world_size)
                max_count = send_counts.max()

                offsets = torch.cumsum(send_counts, dim=0) - send_counts

                positions = torch.arange(sorted_hidden.size(0), device=hidden_states.device)
                local_pos = positions - offsets[global_sorted_ranks]

                hidden_pad = torch.zeros(
                    (self.world_size, max_count, hidden_dim),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )

                index_pad = torch.zeros(
                    (self.world_size, max_count),
                    dtype=torch.long,
                    device=hidden_states.device
                )

                weight_pad = torch.zeros(
                    (self.world_size, max_count),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                hidden_pad[global_sorted_ranks, local_pos] = sorted_hidden
                index_pad[global_sorted_ranks, local_pos] = sorted_index
                weight_pad[global_sorted_ranks, local_pos] = sorted_weights
                
                count_split = [
                    torch.tensor([send_counts[r], max_count],
                                 dtype=torch.long,
                                 device=hidden_states.device)
                    for r in range(self.world_size)
                ]

                hidden_split = list(hidden_pad.unbind(0))
                index_split = list(index_pad.unbind(0))
                weight_split = list(weight_pad.unbind(0))

                gather_hidden = [
                    torch.empty((max_count, hidden_dim),
                                dtype=hidden_states.dtype,
                                device=hidden_states.device)
                    for _ in range(self.world_size)
                ]
                
                del hidden_pad, index_pad, weight_pad

            else:
                count_split = None
                hidden_split = None
                index_split = None
                weight_split = None
                gather_hidden = None

            dist.scatter(recv_info,
                         scatter_list=count_split,
                         src=self.main_rank,
                         group=self.moe_group)

            local_count = recv_info[0].item()
            max_count = recv_info[1].item()

            local_hidden_states = torch.empty(
                (max_count, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            ) if self.main_rank == self.rank else hidden_states[:max_count]

            local_k_index = torch.empty(
                (max_count,),
                dtype=torch.long,
                device=hidden_states.device
            )

            local_k_weights = torch.empty(
                (max_count,),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

            dist.scatter(local_hidden_states,
                         scatter_list=hidden_split,
                         src=self.main_rank,
                         group=self.moe_group)

            dist.scatter(local_k_index,
                         scatter_list=index_split,
                         src=self.main_rank,
                         group=self.moe_group)

            dist.scatter(local_k_weights,
                         scatter_list=weight_split,
                         src=self.main_rank,
                         group=self.moe_group)
            
            if self.rank == self.main_rank:
                del hidden_split, index_split, weight_split
            
            local_k_index = self.global_to_local_expert_idxs[local_k_index]

            expert_mask = torch.nn.functional.one_hot(
                local_k_index[:local_count],
                num_classes=len(self.local_experts)
            ).transpose(0, 1).to(torch.bool)

            local_hidden_states[:local_count] = self.compute(local_hidden_states[:local_count], expert_mask)

            local_hidden_states[:local_count].mul_(local_k_weights[:local_count].unsqueeze(-1))

            dist.gather(local_hidden_states,
                        gather_list=gather_hidden,
                        dst=self.main_rank,
                        group=self.moe_group)

            if self.rank == self.main_rank:
                gathered = torch.cat([hidden[:recv_info[0].item()] 
                                      for hidden, recv_info in zip(gather_hidden, count_split)],
                                     dim=0)
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
                 moe_group: ProcessGroup,
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
        token_indices = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(k)
        flat_index = top_k_index.reshape(-1)
        flat_weights = top_k_weights.reshape(-1)

        recv_info = torch.empty((2,), dtype=torch.long, device=hidden_states.device)

        if self.rank == self.main_rank:
            global_ranks = self.expert_to_rank[flat_index]
            global_sorted_ranks, global_sorted_indices = torch.sort(global_ranks, stable=True)

            sorted_token_indices = token_indices[global_sorted_indices]
            sorted_hidden = hidden_states[sorted_token_indices]
            sorted_index = flat_index[global_sorted_indices]
            sorted_weights = flat_weights[global_sorted_indices]

            send_counts = torch.bincount(global_sorted_ranks, minlength=self.world_size)
            max_count = send_counts.max()

            offsets = torch.cumsum(send_counts, dim=0) - send_counts
            positions = torch.arange(sorted_hidden.size(0), device=hidden_states.device)
            local_pos = positions - offsets[global_sorted_ranks]

            hidden_pad = torch.zeros(
                (self.world_size, max_count, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            ) 

            index_pad = torch.zeros(
                (self.world_size, max_count),
                dtype=torch.long,
                device=hidden_states.device
            )

            weight_pad = torch.zeros(
                (self.world_size, max_count),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

            hidden_pad[global_sorted_ranks, local_pos] = sorted_hidden
            index_pad[global_sorted_ranks, local_pos] = sorted_index
            weight_pad[global_sorted_ranks, local_pos] = sorted_weights
            
            count_split = [
                torch.tensor([send_counts[r], max_count],
                             dtype=torch.long,
                             device=hidden_states.device)
                for r in range(self.world_size)
            ]

            hidden_split = list(hidden_pad.unbind(0))
            index_split = list(index_pad.unbind(0))
            weight_split = list(weight_pad.unbind(0))

            gather_hidden = [
                torch.empty((max_count, hidden_dim),
                            dtype=hidden_states.dtype,
                            device=hidden_states.device)
                for _ in range(self.world_size)
            ]

            del hidden_pad, index_pad, weight_pad

        else:
            count_split = None
            hidden_split = None
            index_split = None
            weight_split = None
            gather_hidden = None
            
        dist.scatter(recv_info,
                     scatter_list=count_split,
                     src=self.main_rank,
                     group=self.moe_group)

        local_count = recv_info[0].item()
        max_count = recv_info[1].item()
        
        local_hidden_states = torch.empty(
            (max_count, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        ) 

        local_k_index = torch.empty(
            (max_count,),
            dtype=torch.long,
            device=hidden_states.device
        ) 

        local_k_weights = torch.empty(
            (max_count,),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        ) 
        dist.scatter(local_hidden_states,
                     scatter_list=hidden_split,
                     src=self.main_rank,
                     group=self.moe_group)
        
        dist.scatter(local_k_index,
                     scatter_list=index_split,
                     src=self.main_rank,
                     group=self.moe_group)
        
        dist.scatter(local_k_weights,
                     scatter_list=weight_split,
                     src=self.main_rank,
                     group=self.moe_group)

        if self.rank == self.main_rank:
            del hidden_split, index_split, weight_split

        local_k_index = self.global_to_local_expert_idxs[local_k_index[:local_count]]

        expert_mask = torch.nn.functional.one_hot(
            local_k_index,
            num_classes=len(self.local_experts)
        ).transpose(0, 1).to(torch.bool)
        local_hidden_states[:local_count] = self.compute(local_hidden_states[:local_count], expert_mask)

        local_hidden_states[:local_count].mul_(local_k_weights[:local_count].unsqueeze(-1))
        dist.gather(local_hidden_states,
                    gather_list=gather_hidden,
                    dst=self.main_rank,
                    group=self.moe_group)
        if self.rank == self.main_rank:
            gathered = torch.cat(
                [gather_hidden[r][:count_split[r][0]] for r in range(self.world_size)],
                dim=0
            ) 

            inverse_indices = torch.empty_like(global_sorted_indices)
            inverse_indices[global_sorted_indices] = torch.arange(
                global_sorted_indices.size(0),
                device=global_sorted_indices.device
            )

            gathered = gathered[inverse_indices]
            gathered = gathered.view(num_tokens, k, hidden_dim)
            output = gathered.sum(dim=1)
            return output

        else:
            return hidden_states