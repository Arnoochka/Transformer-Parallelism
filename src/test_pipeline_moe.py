import torch
import torch.distributed as dist
from mytransformers import utils
from mytransformers.parallel.moe_parallel import moe_pp
from mytransformers.benchmark import moe_test
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
    
    model = moe_test.TestModel(moe_test.Config).eval()
    model = moe_test.TestMoeGenerator(module=model,
                                      scheduler=moe_pp.RoundRobinScheduler(),
                                       num_stages=2,
                                       groups_info=stages,
                                       inner_comm_groups=inner_comm_groups,
                                       final_comm_group=None,
                                       embed_size=moe_test.Config.hidden_size,
                                       vocab_size=moe_test.Config.vocab_size,
                                       device=device)
    utils.Logger.log_all_device(f"MODEL_SIZE:{utils.get_model_size(model):.3f} GB\n{model}")


    with torch.no_grad():
        input_ids = torch.randint(0, moe_test.Config.vocab_size, (2, 16)).to(device)
        mbatches = [moe_pp.MBatch(data={"input_ids": input_ids},
                                  idx=idx,
                                  stream=torch.cuda.Stream(),
                                  event=torch.cuda.Event())
                    for idx in range(4)] 
        outputs: List[moe_pp.MBatch] = model(mbatches, use_cache=True)

        for _ in range(10):
            mbatches = [moe_pp.MBatch(data={"input_ids": out.data['logits'][:, -1:].argmax(dim=-1),
                                            "past_key_values": out.data['past_key_values']},
                                  idx=idx,
                                  stream=torch.cuda.Stream(),
                                  event=torch.cuda.Event())
                        for idx, out in enumerate(outputs)]
            
            outputs: List[moe_pp.MBatch] = model(mbatches, use_cache=True)
            
            
    utils.Logger.log_all_device(f"MEMORY:{torch.cuda.max_memory_allocated() / utils.MemoryUnits.GB.value:.3f} GB")
    