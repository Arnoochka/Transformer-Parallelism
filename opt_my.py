import os
import torch
import torch.distributed as dist
from mytransformers import tp_custom
from mytransformers import utils
from transformers import (AutoTokenizer, OPTForCausalLM)

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    model = utils.init_distributed(model, tp_custom.OPTGenerator)
    utils.logger([child for child in model.named_children()], rank)
    tokens = torch.LongTensor(tokenizer.encode('Input String')).unsqueeze(0).to(torch.cuda.current_device())
    tp_group = utils.TP_GROUP
    local_rank = dist.get_rank(tp_group)
    output = model.generate(tokens, use_cache=False)
    print(f"---device:{local_rank}---\npeak mem: {torch.cuda.max_memory_allocated() / 1024**3:.3f}")
    dist.barrier(tp_group)
    utils.logger(f"output: {tokenizer.decode(output.squeeze(0))}", local_rank)