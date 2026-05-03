import torch
from mytransformers.benchmark import moe_test

def test_sort_efficiency(top_k_index: torch.Tensor, 
                         permutation: torch.Tensor, 
                         expert_to_rank: torch.Tensor, 
                         world_size: int):
    """
    Рассчитывает точность распределения и реальную экономию трафика AllToAll (Volume-based).
    """
    num_tokens, k = top_k_index.size()
    chunk_size = num_tokens // world_size
    device = top_k_index.device
    
    # 1. Определяем финальные ранги после сортировки
    assigned_ranks = permutation // chunk_size  # [num_tokens]
    
    # 2. Ранги экспертов, которые нужны каждому токену
    desired_ranks = expert_to_rank[top_k_index]  # [num_tokens, k]
    
    # --- МЕТРИКА 1: HIT RATE (Локальность) ---
    # Попал, если хотя бы один эксперт оказался на том же GPU, что и токен
    hits_mask = (assigned_ranks.unsqueeze(1) == desired_ranks).any(dim=1)
    num_hits = hits_mask.sum().item()
    hit_rate = num_hits / num_tokens

    # --- МЕТРИКА 2: СЕТЕВОЙ ТРАФИК (Volume) ---
    # Для каждого токена считаем: k - (сколько экспертов на его текущем ранге)
    # Это и есть количество "посылок" через AllToAll для этого токена
    is_local = (assigned_ranks.unsqueeze(1) == desired_ranks) # [num_tokens, k]
    network_copies_per_token = k - is_local.sum(dim=1)
    total_volume_sort = network_copies_per_token.sum().item()

    # --- СРАВНЕНИЕ С НАИВНЫМ ПОДХОДОМ ---
    initial_ranks = torch.arange(num_tokens, device=device) // chunk_size
    
    # Наивный Hit Rate
    naive_hits_mask = (initial_ranks.unsqueeze(1) == desired_ranks).any(dim=1)
    naive_hit_rate = naive_hits_mask.sum().item() / num_tokens
    
    # Наивный сетевой трафик
    naive_is_local = (initial_ranks.unsqueeze(1) == desired_ranks)
    naive_volume = (k - naive_is_local.sum(dim=1)).sum().item()
    
    # Расчет экономии трафика (Volume Reduction)
    if naive_volume > 0:
        traffic_reduction = ((naive_volume - total_volume_sort) / naive_volume) * 100
    else:
        traffic_reduction = 0.0

    print(f"--- Результаты теста (Uniform, k={k}) ---")
    print(f"Токенов: {num_tokens} | Ranks: {world_size}")
    print(f"Hit Rate (хотя бы 1 эксперт локально):")
    print(f"  > Sort:  {hit_rate * 100:.2f}%")
    print(f"  > Naive: {naive_hit_rate * 100:.2f}%")
    print(f"--------------------------------")
    print(f"Сетевой объем (копий токенов в AllToAll):")
    print(f"  > Sort:  {total_volume_sort}")
    print(f"  > Naive: {naive_volume}")
    print(f"ЭКОНОМИЯ ТРАФИКА: {traffic_reduction:.2f}%")
    print(f"Среднее число пересылок на токен: {total_volume_sort / num_tokens:.3f}")
 
    return {
        "hit_rate": hit_rate,
        "traffic_reduction_pct": traffic_reduction,
        "total_volume_sort": total_volume_sort,
        "total_volume_naive": naive_volume
    }

# ==========================================
# Синтетический тест для проверки логики
# ==========================================
if __name__ == "__main__":
    # Параметры теста
    num_tokens = 32
    world_size = 2
    num_experts = 32
    k = 2
    device = torch.device("cpu") # Можно заменить на "cuda"

    # 1. Генерируем случайный маппинг экспертов по рангам (по 2 эксперта на ранг)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    expert_to_rank = torch.arange(num_experts, device=device) // (num_experts // world_size)
    
    # 2. Генерируем случайные запросы токенов к экспертам
    top_k_index, _ = moe_test.TokenRouter.uniform(router_logits, k)
    
    # 3. Мокаем класс, чтобы запустить твой метод sort
    class MockMoE:
        def __init__(self):
            self.moe_group = None
            class Experts:
                pass
            self.experts = Experts()
            self.experts.expert_to_rank = expert_to_rank
            
        # Вставляем сюда твои функции sort и assign_residual_tokens
        # (я опустил dist.get_world_size и заменил на world_size напрямую для теста)
        def sort(self, top_k_index):
            chunk_size = num_tokens // world_size
            permutation = torch.empty(num_tokens, dtype=torch.long, device=top_k_index.device)
            
            expert_ranks = self.experts.expert_to_rank[top_k_index]
            
            rank_counts = torch.zeros((num_tokens, world_size), dtype=torch.long, device=top_k_index.device)
            rank_counts.scatter_add_(dim=1, index=expert_ranks, src=torch.ones_like(expert_ranks))
            
            target_ranks = torch.argmax(rank_counts, dim=1)
            
            one_hot_target_ranks = torch.nn.functional.one_hot(target_ranks, num_classes=world_size).to(torch.long)
            pos_rank = torch.cumsum(one_hot_target_ranks, dim=0) - 1
            pos_rank = pos_rank.gather(dim=1, index=target_ranks.unsqueeze(1)).squeeze(1)
            
            valid = pos_rank < chunk_size
            valid_tokens = torch.nonzero(valid).squeeze(1)
            valid_ranks = target_ranks[valid_tokens]
            valid_pos = pos_rank[valid_tokens]
            
            rank_offset = torch.arange(world_size, device=top_k_index.device) * chunk_size
            permutation[valid_tokens] = rank_offset[valid_ranks] + valid_pos
            permutation[~valid] = -1
            
            permutation = self.assign_residual_tokens(permutation, rank_counts, chunk_size, world_size, min(world_size, k))
            return permutation
            
        def assign_residual_tokens(self, permutation, rank_counts, chunk_size, world_size, num_round):
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
                global_pos = (bid_rank[accepted_local] * chunk_size + next_pos[bid_rank[accepted_local]] + pos_round[accepted_local])
                permutation[rest_idx[accepted_local]] = global_pos
                placed = placed | accepted
                
                delta = torch.zeros(world_size, dtype=torch.long, device=permutation.device)
                delta.scatter_add_(dim=0, index=bid_rank[accepted_local], src=torch.ones_like(accepted_local))
                next_pos = next_pos + delta
                remaining_cap = remaining_cap - delta
            
            if not placed.all():
                placed_global_mask = permutation != -1
                used = torch.zeros(num_tokens, dtype=torch.bool, device=permutation.device)
                used[permutation[placed_global_mask]] = True
                free_positions = torch.nonzero(~used).squeeze(1)
                unplaced_idx = torch.nonzero(~placed_global_mask).squeeze(1)
                permutation[unplaced_idx] = free_positions
            
            return permutation

    # Инициализируем мок и запускаем
    model = MockMoE()
    permutation = model.sort(top_k_index)
    
    # Тестируем точность
    stats = test_sort_efficiency(top_k_index, permutation, expert_to_rank, world_size)