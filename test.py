import os
import torch
from mytransformers import utils
from mytransformers import pp, pp_custom
from transformers import (AutoTokenizer, OPTForCausalLM)
from mytransformers import bench
from torch.distributed import barrier

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    utils.init_distributed_cuda()
    first_stage = [utils.create_group([0]), [0]]
    second_stage = [utils.create_group([1]), [1]]
    pp_custom.OPTGenerator.groups_info = [first_stage, second_stage]
    benchmark = bench.BenchmarkModel(
        model,
        pp_custom.OPTGenerator,
        tokenizer,
        model_name="my_opt-1.3b")
    promts = utils.get_prompts("/home/victor/Transformer-Parallelism/Data/benchmark_mini.txt")
    results = benchmark(promts, utils.GROUP)
    utils.Logger.log_main_device(results, rank)
    barrier(utils.GROUP)
    utils.Logger.log_all_device(model)
    
    