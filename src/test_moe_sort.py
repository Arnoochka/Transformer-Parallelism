import torch
import pandas as pd
import numpy as np
from mytransformers.benchmark import moe_test


# Твой класс MockMoE без изменений логики
class MockMoE:
    def __init__(self, expert_to_rank, world_size, num_tokens):
        self.experts = type('Experts', (), {'expert_to_rank': expert_to_rank})()
        self.world_size = world_size
        self.num_tokens = num_tokens

    def sort(self, top_k_index):
        chunk_size = self.num_tokens // self.world_size
        permutation = torch.empty(self.num_tokens, dtype=torch.long, device=top_k_index.device)
        expert_ranks = self.experts.expert_to_rank[top_k_index]
        rank_counts = torch.zeros((self.num_tokens, self.world_size), dtype=torch.long, device=top_k_index.device)
        rank_counts.scatter_add_(dim=1, index=expert_ranks, src=torch.ones_like(expert_ranks))
        target_ranks = torch.argmax(rank_counts, dim=1)
        one_hot_target_ranks = torch.nn.functional.one_hot(target_ranks, num_classes=self.world_size).to(torch.long)
        pos_rank = torch.cumsum(one_hot_target_ranks, dim=0) - 1
        pos_rank = pos_rank.gather(dim=1, index=target_ranks.unsqueeze(1)).squeeze(1)
        valid = pos_rank < chunk_size
        valid_tokens = torch.nonzero(valid).squeeze(1)
        valid_ranks = target_ranks[valid_tokens]
        valid_pos = pos_rank[valid_tokens]
        rank_offset = torch.arange(self.world_size, device=top_k_index.device) * chunk_size
        permutation[valid_tokens] = rank_offset[valid_ranks] + valid_pos
        permutation[~valid] = -1
        self.assign_residual_tokens(permutation, rank_counts, chunk_size, self.world_size, min(self.world_size, k))
        return permutation
            
    def assign_residual_tokens(self, permutation, rank_counts, chunk_size, world_size, num_round):
        num_tokens = permutation.size(0)
        rest_mask = (permutation == -1)
        rest_idx = torch.nonzero(rest_mask).squeeze(1)
        num_res_tokens = rest_idx.size(0)
        if num_res_tokens == 0: return permutation
        placed_ranks_stage1 = permutation[~rest_mask] // chunk_size
        occupied = torch.bincount(placed_ranks_stage1, minlength=world_size)
        next_pos = occupied.clone()
        remaining_cap = chunk_size - occupied
        rest_counts = rank_counts[rest_idx]
        options_mask = rest_counts > 0
        options = torch.argsort(options_mask.to(torch.int8), dim=1, descending=True, stable=True)[:, :num_round]
        option_valid = options_mask.gather(dim=1, index=options)
        placed = torch.zeros(num_res_tokens, dtype=torch.bool, device=permutation.device)
        for i in range(num_round):
            active = (~placed) & option_valid[:, i]
            if not active.any(): break
            bid_rank = options[:, i]
            one_hot_bid_rank = torch.nn.functional.one_hot(bid_rank, num_classes=world_size) * active.to(torch.long).unsqueeze(1)
            pos_round = (torch.cumsum(one_hot_bid_rank, dim=0) - 1).gather(dim=1, index=bid_rank.unsqueeze(1)).squeeze(1)
            accepted = active & (pos_round < remaining_cap[bid_rank])
            accepted_local = torch.nonzero(accepted).squeeze(1)
            global_pos = (bid_rank[accepted_local] * chunk_size + next_pos[bid_rank[accepted_local]] + pos_round[accepted_local])
            permutation[rest_idx[accepted_local]] = global_pos
            placed = placed | accepted
            delta = torch.zeros(world_size, dtype=torch.long, device=permutation.device)
            delta.scatter_add_(dim=0, index=bid_rank[accepted_local], src=torch.ones_like(accepted_local))
            next_pos += delta
            remaining_cap -= delta
        if not placed.all():
            used = torch.zeros(num_tokens, dtype=torch.bool, device=permutation.device)
            used[permutation[permutation != -1]] = True
            permutation[permutation == -1] = torch.nonzero(~used).squeeze(1)
        return permutation

def calculate_metrics(top_k_index, permutation, expert_to_rank, world_size):
    num_tokens, k = top_k_index.size()
    chunk_size = num_tokens // world_size
    device = top_k_index.device
    
    # Ранги экспертов для каждого токена
    desired_ranks = expert_to_rank[top_k_index.long()]
    
    # --- МЕТРИКИ SORT ---
    assigned_ranks = permutation // chunk_size
    hits_mask = (assigned_ranks.unsqueeze(1) == desired_ranks).any(dim=1)
    hit_rate_sort = hits_mask.sum().item() / num_tokens

    is_local = (assigned_ranks.unsqueeze(1) == desired_ranks)
    total_volume_sort = (k - is_local.sum(dim=1)).sum().item()

    # --- МЕТРИКИ NAIVE (Изначальное распределение) ---
    initial_ranks = torch.arange(num_tokens, device=device) // chunk_size
    naive_hits_mask = (initial_ranks.unsqueeze(1) == desired_ranks).any(dim=1)
    hit_rate_naive = naive_hits_mask.sum().item() / num_tokens
    
    naive_is_local = (initial_ranks.unsqueeze(1) == desired_ranks)
    naive_volume = (k - naive_is_local.sum(dim=1)).sum().item()
    
    reduction = ((naive_volume - total_volume_sort) / naive_volume * 100) if naive_volume > 0 else 0
    
    return {
        "hit_rate_sort": hit_rate_sort,
        "hit_rate_naive": hit_rate_naive, # Добавили сюда
        "traffic_reduction_pct": reduction,
        "avg_transfers_sort": total_volume_sort / num_tokens,
        "avg_transfers_naive": naive_volume / num_tokens
    }

# Функция запуска одного эксперимента
def run_experiment(num_tokens, world_size, num_experts, k):
    device = torch.device("cpu")
    expert_to_rank = torch.arange(num_experts, device=device) // (num_experts // world_size)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    top_k_index, _ = moe_test.TokenRouter.uniform(router_logits, k)
    
    model = MockMoE(expert_to_rank, world_size, num_tokens)
    permutation = model.sort(top_k_index)
    
    metrics = calculate_metrics(top_k_index, permutation, expert_to_rank, world_size)
    metrics.update({"k": k, "world_size": world_size, "num_tokens": num_tokens})
    return metrics

if __name__ == "__main__":
    num_tokens = 32768
    num_experts = 64
    
    k_values = [1, 2, 4, 8]
    world_size_values = [2, 4, 8, 16, 32, 64]
    iterations = 10
    
    results = []
    
    print("Запуск экспериментов...")
    for k in k_values:
        for ws in world_size_values:
            # Проверка, чтобы chunk_size был целым
            if num_tokens % ws != 0: continue
                
            for i in range(iterations):
                res = run_experiment(num_tokens, ws, num_experts, k)
                results.append(res)
    
    df = pd.DataFrame(results)
    
    # Группируем для красоты
    summary = df.groupby(['k', 'world_size']).mean().reset_index()
    print("\n--- Сводные результаты (среднее) ---")
    cols_to_show = [
        'k', 
        'world_size', 
        'hit_rate_sort', 
        'hit_rate_naive', 
        'traffic_reduction_pct', 
        'avg_transfers_sort'
    ]
    
    print(summary[cols_to_show])
    summary.to_csv("moe_sort_all.csv", index=False)