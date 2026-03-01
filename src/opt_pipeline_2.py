import os
import torch
import torch.distributed as dist
from typing import List
from mytransformers import utils
from mytransformers import pp_custom_2 as pp_custom
from mytransformers.parallel import pp_2 as pp
from transformers import AutoTokenizer, OPTForCausalLM
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
        batches.append(pp.MBatch(data=inputs,
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
    generate_func=GenerationFunc.pipeline_generate,
    batch_func=pipeline_batch_func,
    warm_up=True,
    model_name="opt-1.3b-pipeline",
    description="Pipeline parallel OPT-1.3B benchmark",
    max_prompt_len=max_prompt_len,
    max_new_tokens=max_new_tokens,
    dtype=torch.float32,
    save_model_config=False,
    save_stats=True,
    save_dir=f"results/opt/pipeline_2/batch_size={batch_size}-num_microbatch={num_microbatches}-max_prompt_len={max_prompt_len}-max_new_tokens={max_new_tokens}")
    stats = benchmark(
    prompts=prompts,
    batch_size=batch_size // num_microbatches,
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

    model = OPTForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    ).eval()

    stages = [
        (utils.create_group([0]), [0]),
        (utils.create_group([1]), [1]),
    ]

    comm_groups = [
        utils.create_group([0, 1]),
        utils.create_group([0, 1])
    ]

    pp_custom.OPTGenerator(
        module=model,
        num_stages=2,
        groups_info=stages,
        comm_groups=comm_groups,
        embed_size=2048,
        vocab_size=50272,
        device=device
    )
    with open('test.txt', 'r', encoding='utf-8') as file:
        text = file.read()
        
    for batch_size in range(56, 64 + 1, 8):
        prompts = [text for _ in range(batch_size)]
        for max_prompt_len in range(16, 112 + 1, 32):
            for max_new_tokens in range(16, 112 + 1, 32):
                for num_microbatches in [1, 2, 4, 8]:
                    start(prompts, batch_size, num_microbatches, max_prompt_len, max_new_tokens)
