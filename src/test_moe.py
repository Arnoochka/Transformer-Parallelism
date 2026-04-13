import torch
from torch import nn
<<<<<<< HEAD
from torch import Tensor
=======
from torch import Tensor, LongTensor
>>>>>>> main
from mytransformers.parallel import moe
from mytransformers import utils
from mytransformers.utils import Logger
import torch.distributed as dist
from torch import cuda
import os
from time import time
import pandas as pd
<<<<<<< HEAD
import numpy as np
import random

GB = 1024**3
pd.set_option('display.max_colwidth', None)


class Config:
    num_experts_per_tok = 1
    hidden_size = 2048
    num_experts = 16
    intermediate_size = 4096


=======

GB = 1024**3

pd.set_option('display.max_colwidth', None)

class Config:
    num_experts_per_tok = 5
    hidden_size = 2048
    num_experts = 8
    intermediate_size = 4096

>>>>>>> main
class MixtralMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
<<<<<<< HEAD
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
=======

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

>>>>>>> main
        self.act_fn = nn.ReLU()

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

<<<<<<< HEAD

=======
>>>>>>> main
class MixtralExperts(nn.ModuleList):
    def __init__(self, config: Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        for _ in range(self.num_experts):
            self.append(MixtralMLP(config))

<<<<<<< HEAD
    def forward(self, hidden_states, top_k_index, top_k_weights):
=======
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
>>>>>>> main
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states
<<<<<<< HEAD


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

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.gate(hidden_states)
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states
=======
    
>>>>>>> main
    
# class MixtralSparseMoeBlock(nn.Module):
#     def __init__(self, config: Config):
#         super().__init__()
#         self.top_k = config.num_experts_per_tok
#         self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
#         self.experts = MixtralExperts(config)

#     def route_tokens_to_experts(self, router_logits):
#         routing_weights = torch.nn.functional.softmax(router_logits.float(), dim=-1)
#         top_k_weights, top_k_index = torch.topk(routing_weights, self.top_k, dim=-1)
#         top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
<<<<<<< HEAD
#         top_k_index = torch.randint(
#             low=0,
#             high=8,
#             size=top_k_index.size(),
#             device=router_logits.device
#         )
#         return top_k_index, top_k_weights.to(router_logits.dtype)
    
=======
#         return top_k_index, top_k_weights.to(router_logits.dtype)

>>>>>>> main
#     def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         batch_size, sequence_length, hidden_dim = hidden_states.shape
#         hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
#         router_logits = self.gate(hidden_states)
#         top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
#         hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
#         hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
#         return hidden_states
<<<<<<< HEAD

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # для multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gather_peak_memory_all_ranks(device) -> dict:
    """
    Собирает peak GPU memory (в ГБ) со всех рангов.
    Возвращает словарь вида {"peak_memory_gpu_0": ..., "peak_memory_gpu_1": ..., ...}
    """
    local_mem = torch.tensor(
        [cuda.max_memory_allocated(device) / GB],
        dtype=torch.float32,
        device=device,
    )
    world_size = dist.get_world_size()
    all_mem = [torch.zeros_like(local_mem) for _ in range(world_size)]
    dist.all_gather(all_mem, local_mem)
    return {f"peak_memory_gpu_{i}": round(all_mem[i].item(), 4) for i in range(world_size)}


def record_result(results: list, exp_type: str, batch_size: int, seq_len: int,
                  elapsed: float, peak_mem_per_gpu: dict):
    """Добавляет строку результата в список results (только на rank 0)."""
    if dist.get_rank() != 0:
        return
    throughput = round(batch_size * seq_len / elapsed, 2)
    row = {
        "type": exp_type,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "throughput_tokens_per_s": throughput,
        "time_s": round(elapsed, 4),
        **peak_mem_per_gpu,  
    }
    results.append(row)


def test_dp_memory_parallel(model: MixtralSparseMoeBlock, local_input: Tensor,
                             batch_size: int, seq_len: int, results: list):
    expert_idxs = torch.arange(0, Config.num_experts)
    expert_idxs = list(torch.split(expert_idxs, expert_idxs.size(0) // dist.get_world_size()))
    _gate = model.gate.to(device=torch.cuda.current_device())
    _route = model.route_tokens_to_experts
    gate = lambda hs: _route(_gate(hs))
    buffer = torch.empty_like(torch.split(local_input, local_input.size(0) // dist.get_world_size(), dim=0)[0])
    model = moe.MoeSparseBlockDPModuleGenerator(model, gate, 0, buffer, moe.MoeDPExpertsMemory, expert_idxs, moe_group, device).to(device)
    cuda.reset_max_memory_allocated(device)
    model = model.to(device)
    start = time()
    local_output = model(local_input)
    elapsed = time() - start
    peak_mem = gather_peak_memory_all_ranks(device)
    Logger.log_main_device(f"DP MEMORY: time: {round(elapsed, 3)}, memory: {peak_mem}")
    record_result(results, "dp_memory", batch_size, seq_len, elapsed, peak_mem)


def test_dp_speed_parallel(model: MixtralSparseMoeBlock, local_input: Tensor,
                            batch_size: int, seq_len: int, results: list):
    expert_idxs = torch.arange(0, Config.num_experts)
    expert_idxs = list(torch.split(expert_idxs, expert_idxs.size(0) // dist.get_world_size()))
    _gate = model.gate.to(device=torch.cuda.current_device())
    _route = model.route_tokens_to_experts
    gate = lambda hs: _route(_gate(hs))
    buffer = torch.empty_like(torch.split(local_input, local_input.size(0) // dist.get_world_size(), dim=0)[0])
    model = moe.MoeSparseBlockDPModuleGenerator(model, gate, 0, buffer, moe.MoeDPExpertsSpeed, expert_idxs, moe_group, device).to(device)
    cuda.reset_max_memory_allocated(device)
    model = model.to(device)
    start = time()
    local_output = model(local_input)
    elapsed = time() - start
    peak_mem = gather_peak_memory_all_ranks(device)
    Logger.log_main_device(f"DP SPEED: time: {round(elapsed, 3)}, memory: {peak_mem}")
    record_result(results, "dp_speed", batch_size, seq_len, elapsed, peak_mem)


def test_pp_memory_parallel(model: MixtralSparseMoeBlock, local_input: Tensor,
                             batch_size: int, seq_len: int, results: list):
    expert_idxs = torch.arange(0, Config.num_experts)
    expert_idxs = list(torch.split(expert_idxs, expert_idxs.size(0) // dist.get_world_size()))
    _gate = model.gate.to(device=torch.cuda.current_device())
    _route = model.route_tokens_to_experts
    gate = lambda hs: _route(_gate(hs))
    model = moe.MoeSparseBlockPipeModuleGenerator(model, gate, 0, moe.MoePipeExpertsMemory, expert_idxs, moe_group, device).to(device)
    cuda.reset_max_memory_allocated(device)
    model = model.to(device)
    start = time()
    local_output = model(local_input)
    elapsed = time() - start
    peak_mem = gather_peak_memory_all_ranks(device)
    Logger.log_main_device(f"PIPE MEMORY: time: {round(elapsed, 3)}, memory: {peak_mem}")
    record_result(results, "pp_memory", batch_size, seq_len, elapsed, peak_mem)


def test_pp_speed_parallel(model: MixtralSparseMoeBlock, local_input: Tensor,
                            batch_size: int, seq_len: int, results: list):
    expert_idxs = torch.arange(0, Config.num_experts)
    expert_idxs = list(torch.split(expert_idxs, expert_idxs.size(0) // dist.get_world_size()))
    _gate = model.gate.to(device=torch.cuda.current_device())
    _route = model.route_tokens_to_experts
    gate = lambda hs: _route(_gate(hs))
    model = moe.MoeSparseBlockPipeModuleGenerator(model, gate, 0, moe.MoePipeExpertsSpeed, expert_idxs, moe_group, device).to(device)
    cuda.reset_max_memory_allocated(device)
    model = model.to(device)
    start = time()
    local_output = model(local_input)
    elapsed = time() - start
    peak_mem = gather_peak_memory_all_ranks(device)
    Logger.log_main_device(f"PIPE SPEED: time: {round(elapsed, 3)}, memory: {peak_mem}")
    record_result(results, "pp_speed", batch_size, seq_len, elapsed, peak_mem)


if __name__ == "__main__":
    SEED = 42
    rank = int(os.environ["RANK"])
    moe_group = utils.init_distributed_cuda()
    device = torch.cuda.current_device()
    results = []
    for batch_size in [128, 256, 512, 1024]:
        for seq_len in [1]:
            Logger.log_main_device(f"TEST: batch size = {batch_size}, seq len = {seq_len}")
            for test in [test_dp_memory_parallel, test_dp_speed_parallel, test_pp_memory_parallel, test_pp_speed_parallel]:
                set_seed(SEED)
                local_input = torch.randn(batch_size, seq_len, Config.hidden_size).to(device)
                model = MixtralSparseMoeBlock(Config)
                test(model, local_input, batch_size, seq_len, results)

    if dist.get_rank() == 0:
        df = pd.DataFrame(results)
        base_cols = ["type", "batch_size", "seq_len", "throughput_tokens_per_s", "time_s"]
        gpu_cols = [c for c in df.columns if c.startswith("peak_memory_gpu_")]
        df = df[base_cols + gpu_cols]
        csv_path = "moe_benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nРезультаты сохранены в {csv_path}")
        print(df.to_string())
=======
    
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
        top_k_index = torch.randint(
            low=0,
            high=8,
            size=top_k_index.size(),
            device=router_logits.device
        )
        return top_k_index, top_k_weights.to(router_logits.dtype)
    
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.gate(hidden_states)
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states
    
    
def test_single(model: MixtralSparseMoeBlock, local_input: Tensor):
    cuda.reset_max_memory_allocated(cuda.current_device())
    model = model.to(device)
    input = [torch.empty_like(local_input) for _ in range(2)]
    dist.all_gather(input, local_input, group=moe_group)
    input = torch.cat(input, dim=0)
    start = time()
    local_output = model(local_input)
    end = time()
    Logger.log_all_device(f"SINGLE STATS: time: {round(end-start, 3)}, memory: {round(cuda.max_memory_allocated(cuda.current_device()) / GB, 3)}")
    
def test_parallel(model: MixtralSparseMoeBlock, local_input: Tensor):
    cuda.reset_max_memory_allocated(cuda.current_device())
    expert_idxs = [LongTensor([0,1,2,3]).to(device), LongTensor([4, 5,6,7]).to(device)]
    model.experts = moe.MoeDPExpertsGenerator(model.experts, expert_idxs, moe_group, device).to(device)
    model = model.to(device)
    start = time()
    local_output = model(local_input)
    end = time()
    Logger.log_all_device(f"PARALLEL STATS: time: {round(end-start, 3)}, memory: {round(cuda.max_memory_allocated(cuda.current_device()) / GB, 3)}")
    
if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model = MixtralSparseMoeBlock(Config)
    rank = int(os.environ["RANK"])
    moe_group = utils.init_distributed_cuda()
    device = torch.cuda.current_device()
    local_input = torch.randn(8, 2048, Config.hidden_size).to(device) + (dist.get_rank() + 1) * 7
    test_single(model, local_input)
    test_parallel(model, local_input)
>>>>>>> main
    
