import torch
from torch.distributed import ProcessGroup
from torch import Tensor
from torch.nn import ModuleList
from mytransformers.parallel.ParallelModule import ParallelModule
import torch.distributed as dist
from mytransformers.parallel.moe_parallel.pipeline.Scheduler import BaseScheduler


class MoeExperts(ParallelModule):
    """
    Базовый вычислительный класс Mixture of Experts параллелизма.
    """
    def __init__(self,
                 global_num_experts: int,
                 local_experts: ModuleList,
                 expert_to_rank: Tensor,
                 global_to_local_expert_idxs: Tensor,
                 moe_group: ProcessGroup,
                 scheduler: BaseScheduler):
        super().__init__()
        self.rank = dist.get_rank(group=moe_group)
        self.world_size = dist.get_world_size(group=moe_group)

        self.global_num_experts = global_num_experts
        self.local_experts = local_experts
        self.expert_to_rank = expert_to_rank
        self.global_to_local_expert_idxs = global_to_local_expert_idxs

        self.moe_group = moe_group
        self.scheduler = scheduler
        self.thread_idx = 0

    def compute(self, hidden_states: Tensor, expert_offsets: Tensor) -> Tensor:
        """
        hidden_states предварительно отсортирован по local expert index.
        expert_offsets: (num_local_experts + 1,) — границы каждого эксперта.
        
        hidden_states[start:end] — contiguous view, не копия.
        """
        for expert_idx, expert in enumerate(self.local_experts):
            start = expert_offsets[expert_idx].item()
            end = expert_offsets[expert_idx + 1].item()
            if start == end:
                continue
            expert_output = expert(hidden_states[start:end])
            hidden_states[start:end] = expert_output
            del expert_output
        return hidden_states

    def reset(self) -> None:
        self.thread_idx = 0

    def update_thread_idx(self) -> None:
        self.thread_idx += 1