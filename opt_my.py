import torch
import torch.distributed as dist
from transformers import pipeline
import os
from mytransformers import tp_custom
from mytransformers import utils

if __name__ == "__main__":
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    print(torch.__version__)
    print(torch.cuda.is_available())
    pipe = pipeline("text-generation",
                    model="facebook/opt-350m",
                    device=torch.device('cuda'),
                    torch_dtype=torch.float32)
    pipe.model = utils.init_distributed(pipe.model, tp_custom.OPTGenerator)
    tp_group = utils.TP_GROUP
    local_rank = dist.get_rank(tp_group)
    utils.logger(f"{[module for module in pipe.model.named_children()]}", local_rank)
    output = pipe('Input String')
    print(output)