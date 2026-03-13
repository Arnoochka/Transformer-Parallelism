import os
import torch
import torch.distributed as dist
from typing import List

from transformers import AutoTokenizer, OPTForCausalLM
from torch.distributed.pipelining import pipeline
from torch.distributed.device_mesh import init_device_mesh


from mytransformers import utils
from mytransformers.benchmark import BenchmarkModel, GenerationFunc


# ---------------------------
# HF wrapper -> positional forward
# ---------------------------
class HFWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        return out.logits


# ---------------------------
# Pipeline batch function
# ---------------------------
def pipeline_batch_func(prompts: List[str],
                        batch_size: int,
                        max_prompt_len: int,
                        tokenizer: AutoTokenizer):

    device = torch.cuda.current_device()
    batches = []

    for i in range(0, len(prompts), batch_size):
        texts = prompts[i:i + batch_size]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len
        )

        batches.append((
            inputs.input_ids.to(device),
            inputs.attention_mask.to(device)
        ))

    return batches


# ---------------------------
# Benchmark
# ---------------------------
def start(model, tokenizer, prompts,
          batch_size,
          max_prompt_len,
          max_new_tokens):

    benchmark = BenchmarkModel(
        model=model,
        tokenizer=tokenizer,
        generate_func=GenerationFunc.simple_generate,
        batch_func=pipeline_batch_func,
        warm_up=True,
        model_name="opt-1.3b-pipeline",
        description="Pipeline parallel OPT-1.3B benchmark",
        max_prompt_len=max_prompt_len,
        max_new_tokens=max_new_tokens,
        dtype=torch.float32,
        save_model_config=False,
        save_stats=True,
        save_dir=f"results/opt/pipeline_auto/batch_size={batch_size}-max_prompt_len={max_prompt_len}"
    )

    stats = benchmark(
        prompts=prompts,
        batch_size=batch_size,
        eos_token_id=0,
        pad_token_id=0,
        use_cache=True
    )

    utils.Logger.log_main_device(stats)


# ---------------------------
# MAIN
# ---------------------------
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

    with open('test.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    for batch_size in range(40, 64 + 1, 8):
        prompts = [text for _ in range(batch_size)]

        for max_prompt_len in range(16, 128 + 1, 16):

            # 1. Загружаем HF модель
            hf_model = OPTForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).eval()

            # 2. Создаём device mesh
            mesh = init_device_mesh("cuda", (world_size,))

            # 3. Авто pipeline split
            pipe_model = pipeline(
                hf_model,
                mesh=mesh,
                split_spec={
                    "model.decoder.layers.12": "begin_stage"
                }
            )

            # 4. Wrapper для Benchmark
            model = HFWrapper(pipe_model)

            start(
                model,
                tokenizer,
                prompts,
                batch_size,
                max_prompt_len,
                1
            )