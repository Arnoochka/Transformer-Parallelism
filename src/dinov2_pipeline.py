import os
import torch
import torch.distributed as dist
from typing import List
from mytransformers import utils
from mytransformers import pp_custom
from mytransformers.parallel import pp
from mytransformers.benchmark import BenchmarkModel, GenerationFunc
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import requests


def pipeline_batch_func(prompts: List[Image],
                        batch_size: int,
                        max_prompt_len: int,
                        processor: AutoImageProcessor) -> List:
    device = torch.cuda.current_device()

    batch_id = 0
    batches = []
    for i in range(0, len(prompts), batch_size):
        images = prompts[i:i + batch_size]

        inputs = processor(images=images, return_tensors="pt").to(device)
        batches.append(pp.MBatch(data=inputs,
                                 idx=batch_id,
                                 stream=torch.cuda.Stream(),
                                 event=torch.cuda.Event()))
        batch_id += 1
    return [{"mbatches":batches}]

def start(prompts: List[str],
          batch_size: int,
          num_microbatches: int):
    
    benchmark = BenchmarkModel(
    model=model,
    tokenizer=processor,
    generate_func=GenerationFunc.encode_generate,
    batch_func=pipeline_batch_func,
    warm_up=True,
    model_name="dinov2",
    description="dinov2",
    max_prompt_len=257,
    max_new_tokens=1,
    dtype=torch.float32,
    save_model_config=False,
    save_stats=True,
    save_dir=f"batch_size={batch_size}-num_microbatch={num_microbatches}")
    stats = benchmark(
    prompts=prompts,
    batch_size=batch_size // num_microbatches)
    utils.Logger.log_main_device(stats)
        
if __name__ == "__main__":

    utils.init_distributed_cuda()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.cuda.current_device()

    model_name = "facebook/dinov2-giant"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name).eval().to(device)
    stages = [
        (utils.create_group([0]), [0]),
        (utils.create_group([1]), [1]),
    ]

    inner_comm_groups = [
        utils.create_group([0, 1]),
    ]
    pp_custom.Dinov2Generator(
        module=model,
        num_stages=2,
        groups_info=stages,
        inner_comm_groups=inner_comm_groups,
        final_comm_group=None,
        embed_size=1536,
        vocab_size=0,
        device=device
    )
    image = Image.open(requests.get(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
        stream=True
        ).raw).convert("RGB")
        
    for batch_size in range(8, 8 + 1, 8):
        prompts = [image for _ in range(batch_size)]
        for num_microbatches in [2]:
            start(prompts, batch_size, num_microbatches)
            torch.cuda.empty_cache()