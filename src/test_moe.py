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
import pandas as pd
from typing import List, Callable, Tuple
from mytransformers.benchmark import init_global_tracker
from mytransformers.benchmark.moe import TokenRouter
from mytransformers import moe

GB = 1024**3
pd.set_option('display.max_colwidth', None)
SEED = 42
TOKEN_ROUTER = TokenRouter.uniform
TOKEN_ROUTER.centers = (7, 15)
TOKEN_ROUTER.std = 2.0
TOKEN_ROUTER.alpha = 1.1

class Config:
    num_experts_per_tok = 2
    hidden_size = 2048
    num_experts = 8
    intermediate_size = 8192
    
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

    def route_tokens_to_experts(self, router_logits: Tensor) -> Tuple[Tensor, Tensor]:
        top_k_index, top_k_weights = TOKEN_ROUTER(router_logits, self.top_k)
        return top_k_index, top_k_weights.to(router_logits.dtype)
    
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.gate(hidden_states)
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states


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
    
    
def start_tests(batch_sizes: List[int], seq_lens: List[int], save_path: str) -> None:
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            Logger.log_main_device(f"TEST: batch size = {batch_size}, seq len = {seq_len}")
            for test in [test_dp_memory_parallel, test_dp_speed_parallel, test_pp_memory_parallel, test_pp_speed_parallel]:
                utils.set_seed(SEED)
                local_input = torch.randn(batch_size, seq_len, Config.hidden_size).to(device)
                model = MixtralSparseMoeBlock(Config)
                test(model, local_input, batch_size, seq_len, results)

    if dist.get_rank() == 0:
        df = pd.DataFrame(results)
        base_cols = ["type", "batch_size", "seq_len", "throughput_tokens_per_s", "time_s"]
        gpu_cols = [c for c in df.columns if c.startswith("peak_memory_gpu_")]
        df = df[base_cols + gpu_cols]
        df.to_csv(save_path, index=False)
        print(df.to_string())
        
def get_operation_info(test: Callable,
                       batch_size: int,
                       seq_len: int,
                       save_path: str) -> None:
    tracker = init_global_tracker(sync_func=dist.barrier) 
    utils.set_seed(SEED)
    local_input = torch.randn(batch_size, seq_len, Config.hidden_size).to(device)
    tracker.start()
    model = MixtralSparseMoeBlock(Config)
    test(model, local_input, batch_size, seq_len, results)
    df = tracker.stop() 
    if dist.get_rank() == 0:
        df.to_csv(save_path)
        print(df.to_string())
    Logger.log_main_device(f"\nРезультаты сохранены в {save_path}")


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    moe_group = utils.init_distributed_cuda()
    device = torch.cuda.current_device()
    TOKEN_ROUTER.weights = torch.tensor([0.0] * 4 + [1.0] * 8 + [0.0] * 4 + [0.0] * 4 + [1.0] * 8 + [0.0] * 4, device=device)
    results = []
    get_operation_info(test_dp_speed_parallel, 48, 768, "dp_results.csv")
    get_operation_info(test_dp_speed_parallel, 48, 768, "dp_results.csv")
    # start_tests(batch_sizes=[48], seq_lens=[768], save_path="results.csv")
    # start_tests(batch_sizes=[32, 48], seq_lens=[512, 768], save_path="results.csv")
    
