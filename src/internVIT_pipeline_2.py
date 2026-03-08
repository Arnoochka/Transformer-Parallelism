import os
import torch
import torch.distributed as dist
from typing import List
from mytransformers import utils
from mytransformers import pp_custom_2 as pp_custom
from mytransformers.parallel import pp_2 as pp
from mytransformers.benchmark import BenchmarkModel, GenerationFunc
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import requests
import types
import sys
import importlib

def pipeline_batch_func(prompts: List[Image],
                        batch_size: int,
                        max_prompt_len: int,
                        processor: AutoImageProcessor) -> List:
    device = torch.cuda.current_device()

    batch_id = 0
    batches = []
    for i in range(0, len(prompts), batch_size):
        images = prompts[i:i + batch_size]

        inputs = processor(images=images, return_tensors="pt").to(torch.bfloat16).to(device)
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
    model_name="intern-6b",
    description="intern-6b",
    max_prompt_len=257,
    max_new_tokens=1,
    dtype=torch.bfloat16,
    save_model_config=False,
    save_stats=True,
    save_dir=f"results/intern/pipeline_2/batch_size={batch_size}-num_microbatch={num_microbatches}")
    stats = benchmark(
    prompts=prompts,
    batch_size=batch_size // num_microbatches)
    utils.Logger.log_main_device(stats)
        
if __name__ == "__main__":

    utils.init_distributed_cuda()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.cuda.current_device()

    model_name = "OpenGVLab/InternViT-6B-448px-V2_5"

    processor = AutoImageProcessor.from_pretrained(model_name)
    flash_attn_stub = types.ModuleType("flash_attn")
    flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    sys.modules["flash_attn"] = flash_attn_stub
    original_linspace = torch.linspace
    def patched_linspace(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return original_linspace(*args, **kwargs)

    torch.linspace = patched_linspace
    model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
    ).eval()
    torch.linspace = original_linspace
    
    stages = [
        (utils.create_group([0]), [0]),
        (utils.create_group([1]), [1]),
        (utils.create_group([2]), [2]),
        (utils.create_group([3]), [3]),
    ]

    inner_comm_groups = [
        utils.create_group([0, 1]),
        utils.create_group([1, 2]),
        utils.create_group([2, 3])
        ]

    pp_custom.InternGenerator(
        module=model,
        num_stages=4,
        groups_info=stages,
        inner_comm_groups=inner_comm_groups,
        final_comm_group=None,
        embed_size=3200,
        vocab_size=0,
        device=device
    )
    utils.Logger.log_all_device(model)
    image = Image.open(requests.get(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
        stream=True
        ).raw).convert("RGB")
        
    for batch_size in [16, 24, 40]:
        prompts = [image for _ in range(batch_size)]
        for num_microbatches in [1, 2, 4, 8]:
            start(prompts, batch_size, num_microbatches)
