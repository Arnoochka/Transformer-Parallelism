import os
import torch
from mytransformers import tp_custom
from mytransformers import utils
from mytransformers import bench
from transformers import (AutoTokenizer, OPTForCausalLM)

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    tp_group = utils.init_distributed_cuda()
    tp_custom.set_tp_group(tp_group)
    benchmark = bench.BenchmarkModel(
        model,
        tp_custom.OPTGenerator,
        tokenizer,
        model_name="my_opt-1.3b")
    promts = utils.get_prompts("/home/victor/Transformer-Parallelism/Data/benchmark_mini.txt")
    results = benchmark(promts, tp_group)
    utils.Logger.log_main_device(results)
    