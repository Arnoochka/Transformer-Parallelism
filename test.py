import torch
from torch import nn
from torch import Tensor, LongTensor
from mytransformers.parallel import moe
from mytransformers import utils
from mytransformers.utils import Logger
import torch.distributed as dist
from torch import cuda
import os
from time import time

GB = 1024**3

class Config:
    num_experts_per_tok = 2
    hidden_size = 2048
    num_experts = 8
    intermediate_size = 4096

class MixtralMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = nn.ReLU()

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class MixtralExperts(nn.ModuleList):
    def __init__(self, config: Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        for _ in range(self.num_experts):
            self.append(MixtralMLP(config))

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size * sequence_length, hidden_dim)
            selected_experts: (batch_size * sequence_length, top_k)
            routing_weights: (batch_size * sequence_length, top_k)
        Returns:
            (batch_size * sequence_length, hidden_dim)
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states
    
    
class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = MixtralExperts(config)

    def route_tokens_to_experts(self, router_logits):
        routing_weights = torch.nn.functional.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_index = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights.to(router_logits.dtype)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.gate(hidden_states)
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states

if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model = MixtralSparseMoeBlock(Config)
    rank = int(os.environ["RANK"])
    moe_group = utils.init_distributed_cuda()
    device = torch.cuda.current_device()
    expert_idxs = [LongTensor([0,1,2,3]).to(device), LongTensor([4, 5,6,7]).to(device)]
    moe.MoeDataParallelExpertsGenerator.expert_idxs = expert_idxs
    moe.MoeDataParallelExpertsGenerator.moe_group = moe_group
    local_input = torch.randn(8, 2048, Config.hidden_size).to(device) + (dist.get_rank() + 1) * 7
    input = [torch.empty_like(local_input) for _ in range(2)]
    dist.all_gather(input, local_input, group=moe_group)
    input = torch.cat(input, dim=0)
    
    cuda.reset_max_memory_allocated(cuda.current_device())
    model = model.to(device)
    start = time()
    without_parallel = model(input)
    end = time()
    Logger.log_main_device(f"WITHOUT PARALLEL: {without_parallel}")
    Logger.log_all_device(f"WITHOUT PARALLEL STATS: time: {round(end-start, 3)}, memory: {round(cuda.max_memory_allocated(cuda.current_device()) / GB, 3)}")
    
    model.experts = moe.MoeDataParallelExpertsGenerator(model.experts, device)
    cuda.reset_max_memory_allocated(cuda.current_device())
    model = model.to(device)
    output = model(local_input)
    with_parallel = [torch.empty_like(output) for _ in range(2)]
    dist.all_gather(with_parallel, output, moe_group)
    start = time()
    with_parallel = torch.cat(with_parallel, dim=0)
    end = time()
    Logger.log_main_device(f"WITH PARALLEL: {with_parallel}")
    Logger.log_all_device(f"WITH PARALLEL STATS: time: {round(end-start, 3)}, memory: {round(cuda.max_memory_allocated(cuda.current_device()) / GB, 3)}")
    
    Logger.log_main_device(without_parallel - with_parallel < 10e-3)