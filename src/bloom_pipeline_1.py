import os
import torch
import torch.distributed as dist
from typing import List
from mytransformers import utils
from mytransformers import pp_custom_1 as pp_custom
from mytransformers.parallel import pp_1 as pp
from transformers import AutoTokenizer, BloomForCausalLM
from mytransformers.benchmark import BenchmarkModel, GenerationFunc

def pipeline_batch_func(prompts: List[str],
                        batch_size: int,
                        max_prompt_len: int,
                        tokenizer: AutoTokenizer) -> List:
    device = torch.cuda.current_device()

    batch_id = 0
    batches = []
    for i in range(0, len(prompts), batch_size):
        texts = prompts[i:i + batch_size]

        inputs = tokenizer(texts, return_tensors="pt", max_length=max_prompt_len).to(device)
        batches.append(inputs)
        batch_id += 1
    return batches

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
    model_name="bloom-3b",
    description="Pipeline parallel Bloom-3B benchmark",
    max_prompt_len=max_prompt_len,
    max_new_tokens=max_new_tokens,
    dtype=torch.float32,
    save_model_config=False,
    save_stats=True,
    save_dir=f"results/bloom/pipeline_1/batch_size={batch_size}-max_prompt_len={max_prompt_len}-max_new_tokens={max_new_tokens}")
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

    model_name = "bigscience/bloom-3b"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"
    )

    stages = [
        (utils.create_group([0]), [0]),
        (utils.create_group([1]), [1]),
    ]
    
    with open('test.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    for batch_size in range(8, 16 + 1, 8):
        prompts = [text for _ in range(batch_size)]
        for max_prompt_len in range(8, 24 + 1, 8):
            model = BloomForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32).eval()
            pp_custom.BloomGenerator.batch_size = batch_size
            pp_custom.BloomGenerator.seq_len = max_prompt_len
            pp_custom.BloomGenerator(
                module=model,
                num_stages=2,
                groups_info=stages,
                device=device
            )
            start(prompts, batch_size, max_prompt_len, 1)
