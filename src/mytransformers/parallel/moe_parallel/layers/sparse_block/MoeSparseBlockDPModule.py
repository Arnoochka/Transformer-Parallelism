import torch
import torch.distributed as dist
from torch import Tensor
from mytransformers.parallel.moe_parallel.layers.experts import MoeExperts
from torch.distributed import ProcessGroup
from typing import Callable, Tuple
from mytransformers.parallel.moe_parallel.pipeline.Scheduler import BaseScheduler 
from .MoESparseBlockModule import MoeSparseBlockModule
from mytransformers import utils


class MoeSparseBlockDPModule(MoeSparseBlockModule):
    def __init__(self,
                 experts: MoeExperts,
                 gate: Callable[[Tensor], Tuple[Tensor, Tensor]],
                 k: int,
                 moe_group: ProcessGroup,
                 main_rank: int,
                 next_main_rank: int,
                 scheduler: BaseScheduler):
        super().__init__(experts, gate, k, moe_group, main_rank, next_main_rank, scheduler)
        
    @torch.no_grad()
    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        local_rank = dist.get_rank(self.moe_group)
        world_size = dist.get_world_size(self.moe_group)
        
        hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states.size(0)
        chunk_size = num_tokens // world_size
        
        if local_rank == self.main_rank:
            top_k_index, top_k_weights = self.gate(hidden_states)
        else:
            top_k_index = torch.zeros((num_tokens, self.k),
                                      device=hidden_states.device,
                                      dtype=torch.long)
            top_k_weights = torch.zeros((num_tokens, self.k),
                                        device=hidden_states.device,
                                        dtype=hidden_states.dtype)
            
        
        top_k_index, top_k_weights = self.scheduler.transfer(
            op=self.broadcast_top_k,
            op_info=self.thread_idx,
            op_name="broadcast_top_k",
            top_k_index=top_k_index,
            top_k_weights=top_k_weights)
        
        # permutation = self.sort(top_k_index)
        # inverse = self.get_inverse(permutation)
        
        # if local_rank == self.main_rank:
        #     hidden_states = hidden_states[inverse]
        # top_k_index = top_k_index[inverse]
        # top_k_weights = top_k_weights[inverse]
        splitted = list(torch.split(hidden_states, chunk_size, dim=0))
        local_hidden_states = splitted[local_rank]
        local_top_k_weights = list(torch.split(top_k_weights, chunk_size, dim=0))[local_rank]
        
        self.scheduler.transfer(
            op=dist.scatter,
            op_info=self.thread_idx,
            op_name="scatter",
            tensor=local_hidden_states,
            scatter_list=splitted if local_rank == self.main_rank else None,
            src=dist.get_global_rank(self.moe_group, self.main_rank),
            group=self.moe_group)
        
        output = self.experts(local_hidden_states, top_k_index, local_top_k_weights.to(hidden_states.dtype))
        
        self.scheduler.transfer(
            op=dist.gather,
            op_info=self.thread_idx,
            op_name="gather",
            tensor=output,
            gather_list=splitted if local_rank == self.next_main_rank else None,
            dst=dist.get_global_rank(self.moe_group, self.next_main_rank),
            group=self.moe_group)
        
        # if local_rank == self.next_main_rank:
        #     hidden_states = hidden_states[permutation]
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return hidden_states
    
    def sort(self, top_k_index: Tensor) -> Tensor:
        world_size = dist.get_world_size(self.moe_group)
        num_tokens, k = top_k_index.size()
        chunk_size = num_tokens // world_size
        permutation = torch.empty(num_tokens, dtype=torch.long, device=top_k_index.device)
        
        expert_ranks = self.experts.expert_to_rank[top_k_index]
        
        rank_counts = torch.zeros((num_tokens, world_size),
                                  dtype=torch.long,
                                  device=top_k_index.device)
        rank_counts.scatter_add_(dim=1,
                                 index=expert_ranks,
                                 src=torch.ones_like(expert_ranks))
        
        target_ranks = torch.argmax(rank_counts, dim=1)
        
        # Этап 1: размещаем тех, кому хватило места по argmax-рангу.
        one_hot_target_ranks = torch.nn.functional.one_hot(target_ranks,
                                                           num_classes=world_size).to(torch.long)
        pos_rank = torch.cumsum(one_hot_target_ranks, dim=0) - 1
        pos_rank = pos_rank.gather(dim=1, index=target_ranks.unsqueeze(1)).squeeze(1)
        
        valid = pos_rank < chunk_size
        valid_tokens = torch.nonzero(valid).squeeze(1)
        valid_ranks = target_ranks[valid_tokens]
        valid_pos = pos_rank[valid_tokens]
        
        rank_offset = torch.arange(world_size, device=top_k_index.device) * chunk_size
        permutation[valid_tokens] = rank_offset[valid_ranks] + valid_pos
        permutation[~valid] = -1
        
        # Этап 2: остатки распределяем итеративным аукционом.
        permutation = self.assign_residual_tokens(
            permutation, rank_counts, chunk_size, world_size, min(world_size, k)
        )
        rest_mask = (permutation == -1)
        if rest_mask.any():
            used = torch.zeros(num_tokens, dtype=torch.bool, device=permutation.device)
            used[permutation[~rest_mask]] = True
            free_positions = torch.nonzero(~used).squeeze(1)
            unplaced_idx = torch.nonzero(rest_mask).squeeze(1)
            permutation[unplaced_idx] = free_positions
        
        return permutation
    
    def assign_residual_tokens(
        self,
        permutation: Tensor,
        rank_counts: Tensor,
        chunk_size: int,
        world_size: int,
        num_round: int,
    ) -> Tensor:
        num_tokens = permutation.size(0)
        
        rest_mask = (permutation == -1)
        rest_idx = torch.nonzero(rest_mask).squeeze(1)
        num_res_tokens = rest_idx.size(0)
        if num_res_tokens == 0:
            return permutation
        
        placed_ranks_stage1 = permutation[~rest_mask] // chunk_size
        occupied = torch.bincount(placed_ranks_stage1, minlength=world_size)
        next_pos = occupied.clone()
        remaining_cap = chunk_size - occupied
        
        rest_counts = rank_counts[rest_idx]
        options_mask = rest_counts > 0
        options = torch.argsort(options_mask.to(torch.int8), dim=1, descending=True, stable=True)
        options = options[:, :num_round]
        option_valid = options_mask.gather(dim=1, index=options)
        
        placed = torch.zeros(num_res_tokens, dtype=torch.bool, device=permutation.device)
        
        for i in range(num_round):
            active = (~placed) & option_valid[:, i]
            if not active.any():
                break
            
            bid_rank = options[:, i]
            
            active_long = active.to(torch.long).unsqueeze(1)
            one_hot_bid_rank = torch.nn.functional.one_hot(bid_rank, num_classes=world_size)
            one_hot_bid_rank = one_hot_bid_rank * active_long
            pos_round = torch.cumsum(one_hot_bid_rank, dim=0) - 1
            pos_round = pos_round.gather(dim=1, index=bid_rank.unsqueeze(1)).squeeze(1)
            
            accepted = active & (pos_round < remaining_cap[bid_rank])
            
            accepted_local = torch.nonzero(accepted).squeeze(1)
            global_pos = (bid_rank[accepted_local] * chunk_size
                          + next_pos[bid_rank[accepted_local]]
                          + pos_round[accepted_local])
            permutation[rest_idx[accepted_local]] = global_pos
            placed = placed | accepted
            
            delta = torch.zeros(world_size, dtype=torch.long, device=permutation.device)
            delta.scatter_add_(dim=0,
                               index=bid_rank[accepted_local],
                               src=torch.ones_like(accepted_local))
            next_pos = next_pos + delta
            remaining_cap = remaining_cap - delta
        
        # overflow: непомещённые токены получают любые свободные слоты
        if not placed.all():
            placed_global_mask = permutation != -1
            used = torch.zeros(num_tokens, dtype=torch.bool, device=permutation.device)
            used[permutation[placed_global_mask]] = True
            free_positions = torch.nonzero(~used).squeeze(1)
            unplaced_idx = torch.nonzero(~placed_global_mask).squeeze(1)
            permutation[unplaced_idx] = free_positions
        
        return permutation
    
    def get_inverse(self, permutation: Tensor) -> Tensor:
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.size(0), device=permutation.device)
        return inverse
    
    def broadcast_top_k(self,
                        top_k_index: Tensor,
                        top_k_weights: Tensor) -> Tuple[Tensor, Tensor]:
        src = dist.get_global_rank(self.moe_group, self.main_rank)
        dist.broadcast(top_k_index, src=src, group=self.moe_group)
        dist.broadcast(top_k_weights, src=src, group=self.moe_group)
                
        return top_k_index, top_k_weights
            
                
            