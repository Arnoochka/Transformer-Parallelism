import torch
import torch.distributed as dist
from mytransformers import utils
from mytransformers.benchmark import moe_test
from mytransformers import moe, pp
from typing import List


if __name__ == "__main__":
    
    utils.init_distributed_cuda()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()

    stages = [
        (utils.create_group([0]), [0]),
        (utils.create_group([1]), [1]),
    ]
    inner_comm_groups = [
        utils.create_group([0, 1]),
    ]

    # --- одинаковые веса на всех рангах ---
    utils.set_seed()
    torch.manual_seed(42)
    model = moe_test.TestModel(moe_test.Config).eval().to(device)
    utils.Logger.log_all_device(f"SIMPLE MODEL: {model}")
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # --- одинаковый вход на всех рангах ---
    torch.manual_seed(123)
    input_ids = torch.randint(0, moe_test.Config.vocab_size, (8, 16), device=device)

    # --- прогон ДО генератора ---
    with torch.no_grad():
        ref_logits = model(input_ids=input_ids)['logits']

    # --- применяем генератор ---
    model = moe_test.TestMoePipeGenerator(module=model,
                                          num_stages=2,
                                          groups_info=stages,
                                          inner_comm_groups=inner_comm_groups,
                                          final_comm_group=None,
                                          embed_size=moe_test.Config.hidden_size,
                                          vocab_size=moe_test.Config.vocab_size,
                                          num_experts=moe_test.Config.num_experts,
                                          scheduler=moe.pp.RoundRobinScheduler(),
                                          moe_group=None,
                                          device=device)
    utils.Logger.log_all_device(f"GENERATOR MODEL: {model}")

    # --- прогон ПОСЛЕ генератора: разбиваем на микробатчи, собираем ---
    with torch.no_grad():
        mb_size = 2   # = world_size moe_group
        mbatches = [pp.MBatch(data={"input_ids": input_ids[i:i+mb_size]},
                              idx=i // mb_size,
                              stream=torch.cuda.Stream(),
                              event=torch.cuda.Event())
                    for i in range(0, input_ids.size(0), mb_size)]
        outputs: List[pp.MBatch] = model(mbatches, use_cache=False)

    # --- сравнение (на ранге последней стадии) ---
    if rank == world_size - 1:
        par_logits = torch.cat([o.data['logits'] for o in outputs], dim=0)
        utils.Logger.log_all_device(par_logits)
        utils.Logger.log_all_device(ref_logits)
        max_diff = (ref_logits - par_logits).abs().max().item()
        utils.Logger.log_all_device(f"PAR logits: {par_logits.shape}, sum={par_logits.sum().item():.4f}")
        utils.Logger.log_all_device(f"MAX DIFF: {max_diff:.3e}")
        ref_sum_per_token = ref_logits.sum(dim=-1).flatten()
        par_sum_per_token = par_logits.sum(dim=-1).flatten()

        ref_sorted, _ = ref_sum_per_token.sort()
        par_sorted, _ = par_sum_per_token.sort()

        max_diff_sorted = (ref_sorted - par_sorted).abs().max().item()
        max_diff_raw = (ref_sum_per_token - par_sum_per_token).abs().max().item()

        utils.Logger.log_all_device(f"MAX DIFF raw:    {max_diff_raw:.3e}")
        utils.Logger.log_all_device(f"MAX DIFF sorted: {max_diff_sorted:.3e}")