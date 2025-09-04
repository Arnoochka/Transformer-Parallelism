import os
import torch
import torch.distributed as dist
import torch.nn as nn
from mytransformers import tp
import time


HIDDEN_STATE = 2000
k = 1000

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    tp_group = dist.new_group(ranks=[0, 1], backend="nccl")
    
    linear = nn.Linear(HIDDEN_STATE, HIDDEN_STATE, bias=True)
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        x = torch.randn(2, HIDDEN_STATE).cuda(local_rank)
    else:
        x = torch.zeros(2, HIDDEN_STATE).cuda(local_rank)
        
    dist.broadcast(x, src=0, group=tp_group)
        
    model = linear.cuda(local_rank)
    with torch.no_grad():
        x_1 = model(x)
        if rank == 0:
            print(f"x_1:\n{x_1}")
    
    tp.ColumnParallelLinearGenerator.use_all_gather = True
    model = tp.ColumnParallelLinearGenerator(linear, tp_group)
    x_2 = model(x)
    if rank == 0:
        print(f"x_2:\n{x_2}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
