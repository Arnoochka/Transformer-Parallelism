import os
import torch
from torch.distributed import ProcessGroup
from mytransformers import bench, utils
import deepspeed
from deepspeed import comm
from transformers import AutoTokenizer, OPTForCausalLM, PreTrainedModel

def inference_generator(model: PreTrainedModel, tp_group: ProcessGroup):
    return model.to(torch.cuda.current_device())
        
if __name__ == "__main__":
    deepspeed.init_distributed('nccl')
    rank = comm.get_rank()
    torch.cuda.set_device(rank)
    tp_group = comm.get_world_group()
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    benchmark = bench.BenchmarkModel(
        model,
        inference_generator,
        tokenizer,
        loops=1,
        model_name="deepspeed_single-1.3b")
    promts = utils.get_prompts("/home/victor/Transformer-Parallelism/Data/benchmark_mini.txt")
    results = benchmark(promts, tp_group, print_output_num=3)
    utils.logger(results, rank)
    