import torch
import deepspeed.comm as comm
import deepspeed
from mytransformers import utils
from transformers import (AutoTokenizer, OPTForCausalLM)

if __name__ == "__main__":
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    comm.init_distributed('nccl')
    tp_group = comm.get_world_group()
    world_size = comm.get_world_size(tp_group)
    local_rank = comm.get_rank()
    torch.cuda.set_device(local_rank)
    model = deepspeed.init_inference(model, 
                                     tensor_parallel={"tp_size": world_size},
                                     dtype=torch.float32,
                                     replace_with_kernel_inject=True).to(torch.cuda.current_device())
    tokens = torch.LongTensor(tokenizer.encode('Input String')).unsqueeze(0).to(torch.cuda.current_device())
    output = model.generate(tokens, use_cache=False)
    print(f"---device:{local_rank}---\npeak mem: {torch.cuda.max_memory_allocated() / 1024**3:.3f}")
    comm.barrier(tp_group)
    utils.logger(f"output: {tokenizer.decode(output.squeeze(0))}", local_rank)