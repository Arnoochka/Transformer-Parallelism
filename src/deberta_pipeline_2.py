import os
import torch
import torch.distributed as dist
from typing import List
from mytransformers import utils
from mytransformers import pp_custom_2 as pp_custom
from mytransformers.parallel import pp_2 as pp
from transformers import AutoTokenizer, AutoModel
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
    return [{"mbatches":batches}]

def start(prompts: List[str],
          batch_size: int,
          num_microbatches: int, 
          max_prompt_len: int):
    
    benchmark = BenchmarkModel(
    model=model,
    tokenizer=tokenizer,
    generate_func=GenerationFunc.encode_generate,
    batch_func=pipeline_batch_func,
    warm_up=True,
    model_name="deverta",
    description="deberta",
    max_prompt_len=max_prompt_len,
    max_new_tokens=1,
    dtype=torch.float32,
    save_model_config=False,
    save_stats=True,
    save_dir=f"results/deberta/pipeline_2/batch_size={batch_size}-num_microbatch={num_microbatches}-max_prompt_len={max_prompt_len}")
    stats = benchmark(
    prompts=prompts,
    batch_size=batch_size // num_microbatches)
    utils.Logger.log_main_device(stats)
        
if __name__ == "__main__":

    utils.init_distributed_cuda()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.cuda.current_device()

    model_name = "microsoft/deberta-v2-xxlarge"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"
    )

    model = AutoModel.from_pretrained(
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

    pp_custom.DebertaGenerator(
        module=model,
        num_stages=2,
        groups_info=stages,
        comm_groups=comm_groups,
        embed_size=1536,
        vocab_size=128100,
        device=device
    )
    with open('test.txt', 'r', encoding='utf-8') as file:
        text = file.read()
        
    for batch_size in range(64, 64 + 1, 8):
        prompts = [text for _ in range(batch_size)]
        for max_prompt_len in range(128, 128 + 1, 16):
            for num_microbatches in [1, 2, 4, 8]:
                start(prompts, batch_size, num_microbatches, max_prompt_len)
