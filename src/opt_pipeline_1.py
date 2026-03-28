import os
import torch
import torch.distributed as dist
from typing import List, Dict
from mytransformers import utils
from mytransformers import pp_custom_1 as pp_custom
from transformers import AutoTokenizer, OPTForCausalLM
from mytransformers.benchmark import BenchmarkModel, GenerationFunc

def pipeline_batch_func(prompts: List[str],
                        batch_size: int,
                        max_prompt_len: int,
                        tokenizer: AutoTokenizer) -> Dict:
    device = torch.cuda.current_device()
    batch = tokenizer(prompts, return_tensors="pt", max_length=max_prompt_len).to(device)
    return [batch]

def start(prompts: List[str],
          batch_size: int,
          max_prompt_len: int,
          max_new_tokens: int):
    
    benchmark = BenchmarkModel(
    model=model,
    tokenizer=tokenizer,
    generate_func=GenerationFunc.simple_generate,
    batch_func=pipeline_batch_func,
    warm_up=True,
    model_name="opt-1.3b",
    description="Pipeline parallel OPT-1.3B benchmark",
    max_prompt_len=max_prompt_len,
    max_new_tokens=max_new_tokens,
    dtype=torch.float32,
    save_model_config=False,
    save_stats=True,
    save_dir=f"results/opt/pipeline_1/batch_size={batch_size}-max_prompt_len={max_prompt_len}-max_new_tokens={max_new_tokens}")
    stats = benchmark(
    prompts=prompts,
    batch_size=batch_size,
    eos_token_id=0,
    pad_token_id=0,
    use_cache=True)
    utils.Logger.log_main_device(stats)


if __name__ == "__main__":

    utils.init_distributed_cuda()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.cuda.current_device()

    model_name = "facebook/opt-1.3b"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"
    )

    stages = [
        (utils.create_group([0]), [0]),
        (utils.create_group([1]), [1]),
    ]
    inner_comm_groups = [
        utils.create_group([0, 1]),
        ]
    with open('test.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    for batch_size in [8, 16, 24]:
        prompts = [text for _ in range(batch_size)]
        for max_prompt_len in [128, 192, 256, 320, 384]:
            model = OPTForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32).eval()
            pp_custom.OPTGenerator.batch_size = batch_size
            pp_custom.OPTGenerator.seq_len = max_prompt_len
            pp_custom.OPTGenerator(
                module=model,
                num_stages=2,
                groups_info=stages,
                final_group_info=(utils.create_group([0, 1]), [0, 1]),
                device=device,
                comm_groups=inner_comm_groups,
                final_comm_group=utils.create_group([0, 1])
            )
            utils.Logger.log_all_device(model)
            start(prompts, batch_size, max_prompt_len, 1)
