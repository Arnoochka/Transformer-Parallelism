import os
import torch
from mytransformers import utils
from mytransformers import pp_custom
from transformers import (AutoTokenizer, OPTForCausalLM)

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    utils.init_distributed_cuda()
    first_stage = [utils.create_group([0]), [0]]
    second_stage = [utils.create_group([1]), [1]]
    pp_custom.OPTGenerator.num_stages = 2
    pp_custom.OPTGenerator.groups_info = [first_stage, second_stage]
    pp_custom.OPTGenerator.bcast_groups = [second_stage[0], first_stage[0]]
    pp_custom.OPTGenerator.num_microbatches = 4
    pp_custom.OPTGenerator(model, torch.cuda.current_device())
    utils.Logger.log_all_device(model)
    memory = utils.get_model_size()
    utils.Logger.log_all_device(f"MEMORY: {utils.get_model_size(model)}")
    
    