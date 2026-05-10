import os
import torch
from torch import Tensor
import torch.distributed as dist
from typing import List, Tuple
from mytransformers import utils
from mytransformers.parallel import pp, moe
from transformers import AutoTokenizer
from mytransformers.benchmark import BenchmarkModel, GenerationFunc, moe_test, TokenMetrics, init_global_tracker

PATH = "results/base_test/pipe"

def get_pipe_model(model: moe_test.TestModel,
                   stages: List[Tuple[dist.ProcessGroup, List[int]]],
                   inner_comm_groups: List[dist.ProcessGroup],
                   ) -> moe_test.TestModel:
    
    return moe_test.TestPipeGenerator(module=model,
                                      num_stages=len(stages),
                                      groups_info=stages,
                                      inner_comm_groups=inner_comm_groups,
                                      final_comm_group=None,
                                      embed_size=moe_test.Config.hidden_size,
                                      vocab_size=moe_test.Config.vocab_size,
                                      device=torch.cuda.current_device())
    
def get_moe_model(model: moe_test.TestModel,
                  stages: List[Tuple[dist.ProcessGroup, List[int]]],
                  inner_comm_groups: List[dist.ProcessGroup],
                  ) -> moe_test.TestModel:
    
    return moe_test.TestMoePipeGenerator(module=model,
                                         num_stages=len(stages),
                                         groups_info=stages,
                                         final_comm_group=None,
                                         embed_size=moe_test.Config.hidden_size,
                                         vocab_size=moe_test.Config.vocab_size,
                                         num_experts=moe_test.Config.num_experts,
                                         k=moe_test.Config.num_experts_per_tok,
                                         scheduler=moe.pp.RoundRobinScheduler(),
                                         moe_group=dist.group.WORLD,
                                         device=device)
    
    

def tokenizer(batch_size, seq_len) -> Tensor:
    return torch.randint(0, moe_test.Config.vocab_size, (batch_size, seq_len), device=device)

def pipeline_batch_func(prompts: List[str],
                        batch_size: int,
                        max_prompt_len: int,
                        tokenizer: AutoTokenizer
                        ) -> List[pp.MBatch]:
    device = torch.cuda.current_device()

    batch_id = 0
    batches = []
    for i in range(0, len(prompts), batch_size):
        texts = prompts[i:i + batch_size]

        inputs = tokenizer(len(texts), max_prompt_len).to(device)
        batches.append(pp.MBatch(data={"input_ids": inputs},
                                 idx=batch_id,
                                 stream=torch.cuda.Stream(),
                                 event=torch.cuda.Event()))
        batch_id += 1
    return batches

def start(prompts: List[str],
          batch_size: int,
          num_microbatches: int, 
          max_prompt_len: int,
          max_new_tokens: int):
    
    benchmark = BenchmarkModel(
    model=model,
    tokenizer=tokenizer,
    generate_func=GenerationFunc.test_generate,
    batch_func=pipeline_batch_func,
    warm_up=True,
    model_name="test_model",
    description="test_model",
    max_prompt_len=max_prompt_len,
    max_new_tokens=max_new_tokens,
    dtype=torch.bfloat16,
    save_model_config=False,
    save_stats=True,
    save_dir=f"{PATH}/batch_size={batch_size}-prompt_len={max_prompt_len}-new_tokens={max_new_tokens}-num_microbatch={num_microbatches}-")
    stats = benchmark(
    prompts=prompts,
    batch_size=batch_size // num_microbatches,
    token_metric=TokenMetrics.output_tokens,
    eos_token_id=0,
    pad_token_id=0,
    use_cache=True)
    utils.Logger.log_main_device(stats)


if __name__ == "__main__":
    utils.init_distributed_cuda()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.cuda.current_device()

    stages = [
        (utils.create_group([0]), [0]),
        (utils.create_group([1]), [1]),
        (utils.create_group([2]), [2]),
        (utils.create_group([3]), [3])
    ]

    inner_comm_groups = [
        utils.create_group([0, 1]),
        utils.create_group([1, 2]),
        utils.create_group([2, 3])
        ]
    torch.set_default_dtype(torch.bfloat16)
    model = moe_test.TestModel(moe_test.Config).eval().to(torch.bfloat16).to(device)
    model = get_pipe_model(model, stages, inner_comm_groups)
    utils.Logger.log_all_device(model)
    
    for batch_size in [32]:
        prompts = ["" for _ in range(batch_size)]
        for max_prompt_len in [1024]:
            for max_new_tokens in [2048, 4096]:
                for num_microbatches in [4]:
                    if not os.path.exists(f"{PATH}/batch_size={batch_size}-prompt_len={max_prompt_len}-new_tokens={max_new_tokens}-num_microbatch={num_microbatches}-test_model_stats.json"):
                        start(prompts, batch_size, num_microbatches, max_prompt_len, max_new_tokens)
                    else:
                        utils.Logger.log_main_device(f"Эксперимент batch_size={batch_size}-prompt_len={max_prompt_len}-new_tokens={max_new_tokens}-num_microbatch={num_microbatches}-test_model_stats.json уже проведен")
                        
    
    