import os
import torch
import torch.distributed as dist
from mytransformers import utils
from mytransformers import pp_custom
from mytransformers.parallel import pp
from transformers import (AutoTokenizer, OPTForCausalLM)
from mytransformers.benchmark import init_global_tracker, GenerationFunc

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    utils.init_distributed_cuda()
    first_stage = [utils.create_group([0]), [0]]
    second_stage = [utils.create_group([1]), [1]]
    comm_groups = [utils.create_group([0, 1]), utils.create_group([0, 1])]
    pp_custom.OPTGenerator(module=model,
                           num_stages=2,
                           groups_info=[first_stage, second_stage],
                           comm_groups=comm_groups,
                           embed_size=2048,
                           vocab_size=50272,
                           device=torch.cuda.current_device())
    utils.Logger.log_all_device(model)
    device = torch.cuda.current_device()
    texts =["Deepspeed is a framework" for _ in range(4)]
    inputs = tokenizer(texts, return_tensors="pt", max_length=256).to(device)
    mbatches = [pp.MBatch(data=inputs,
                               idx=k,
                               stream=torch.cuda.Stream(),
                               event=torch.cuda.Event())
                 for k in range(4)]
    utils.Logger.log_all_device(f"INPUTS: {inputs}")
    TRACKER = init_global_tracker()
    TRACKER.start()
    outputs = GenerationFunc.pipeline_generate(model=model,
                                              mbatches=mbatches,
                                              max_new_tokens=48,
                                              eos_token_id=0,
                                              pad_token_id=0,
                                              use_cache=True)
    df = TRACKER.stop()
    df.to_csv("one-batch-results.csv")
    utils.Logger.log_all_device(f"MODEL MEMORY: {utils.get_model_size(model):.3f}")
    utils.Logger.log_all_device(f"OUTPUTS: {outputs}")