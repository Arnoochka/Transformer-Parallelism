import os
import torch
import torch.distributed as dist
from typing import List
from mytransformers import utils
from mytransformers import pp_custom_1 as pp_custom
from mytransformers.parallel import pp_1 as pp
from mytransformers.benchmark import BenchmarkModel, GenerationFunc
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import requests
import types
import sys 
import importlib

def pipeline_batch_func(prompts: List[ImportError],
                        batch_size: int,
                        max_prompt_len: int,
                        processor: AutoImageProcessor) -> List:
    device = torch.cuda.current_device()

    batch_id = 0
    batches = []
    for i in range(0, len(prompts), batch_size):
        images = prompts[i:i + batch_size]

        inputs = processor(images=images, return_tensors="pt").to(torch.bfloat16).to(device)
        batches.append(inputs)
        batch_id += 1
    return batches


def start(prompts: List[str],
          batch_size: int):
    
    benchmark = BenchmarkModel(
    model=model,
    tokenizer=processor,
    generate_func=GenerationFunc.encode_generate,
    batch_func=pipeline_batch_func,
    warm_up=True,
    model_name="intern-6b",
    description="intern-6b",
    max_prompt_len=1,
    max_new_tokens=1,
    dtype=torch.bfloat16,
    save_model_config=False,
    save_stats=True,
    save_dir=f"results/intern/pipeline_1/batch_size={batch_size}")
    stats = benchmark(
    prompts=prompts,
    batch_size=batch_size)
    utils.Logger.log_main_device(stats)


if __name__ == "__main__":

    utils.init_distributed_cuda()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.cuda.current_device()

    model_name = "OpenGVLab/InternViT-6B-448px-V2_5"

    processor = AutoImageProcessor.from_pretrained(model_name)

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
    image = Image.open(requests.get(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
        stream=True
        ).raw).convert("RGB")

    for batch_size in [16, 24]:
        prompts = [image for _ in range(batch_size)]
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
                trust_remote_code=True,
                torch_dtype=torch.bfloat16).eval()
        torch.linspace = original_linspace
        
        pp_custom.InternGenerator.batch_size = batch_size
        pp_custom.InternGenerator(
                module=model,
                num_stages=4,
                groups_info=stages,
                final_group_info=(utils.create_group([0, 1, 2, 3]), [0, 1, 2, 3]),
                device=device,
                comm_groups=inner_comm_groups,
                final_comm_group=utils.create_group([0, 1, 2, 3])
            )
        utils.Logger.log_all_device(model)
        start(prompts, batch_size)