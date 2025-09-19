import os
import torch
from torch.distributed import ProcessGroup
from mytransformers import bench, utils
import deepspeed
from deepspeed import comm
from transformers import AutoTokenizer, OPTForCausalLM, PreTrainedModel

def inference_generator(model: PreTrainedModel, tp_group: ProcessGroup):
    module = deepspeed.init_inference(model, 
                                    tensor_parallel={"tp_size": 2},
                                    replace_with_kernel_inject=True,
                                    dtype=torch.float32).module
    return module
        
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
        model_name="deepspeed_opt-1.3b")
    promts = utils.get_prompts("/home/victor/Transformer-Parallelism/Data/benchmark_mini.txt")
    results = benchmark(promts, tp_group)
    utils.logger(results, rank)
    