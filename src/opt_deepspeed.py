import os
import torch
import torch.distributed as dist
from typing import List
from mytransformers import utils
from transformers import AutoTokenizer, OPTForCausalLM
from mytransformers.benchmark import BenchmarkModel, GenerationFunc
import deepspeed
from deepspeed import comm

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
    generate_func=GenerationFunc.deepspeed_generate,
    batch_func=pipeline_batch_func,
    warm_up=True,
    model_name="opt-1.3b-pipeline",
    description="Pipeline parallel OPT-1.3B benchmark",
    max_prompt_len=max_prompt_len,
    max_new_tokens=max_new_tokens,
    dtype=torch.float32,
    save_model_config=False,
    save_stats=True,
    save_dir=f"results/opt/pipeline_1/batch_size={batch_size}-max_prompt_len={max_prompt_len}")
    stats = benchmark(
    prompts=prompts,
    batch_size=batch_size,
    eos_token_id=0,
    pad_token_id=0,
    use_cache=True)
    utils.Logger.log_main_device(stats)

def inference_generator(model):
    module = deepspeed.init_inference(model, 
                                    tensor_parallel={"tp_size": 2},
                                    replace_with_kernel_inject=True,
                                    dtype=torch.float32).module
    return module
        
if __name__ == "__main__":
    deepspeed.init_distributed('nccl')
    rank = comm.get_rank()
    torch.cuda.set_device(rank)
    model_name = "facebook/opt-1.3b"
    with open('test.txt', 'r', encoding='utf-8') as file:
        text = file.read()
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    model = inference_generator(model)

    for batch_size in range(8, 64 + 1, 8):
        prompts = [text for _ in range(batch_size)]
        for max_prompt_len in range(16, 128 + 1, 16):
            for max_new_tokens in range(16, 128 + 1, 16):
                start(prompts, batch_size, max_prompt_len, max_new_tokens)
