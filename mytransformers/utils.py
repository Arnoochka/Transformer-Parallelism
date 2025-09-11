import os
from typing import Optional
import torch
from torch import Tensor
from torch.nn import Module
from mytransformers.parallel.tensor_parallel import TPModuleGenerator
import torch.distributed as dist

TP_GROUP = None

def logger(log: str, rank: int) -> None:
    if rank == 0:
        print(log)

def torch_round(tensor: Tensor, decimals: int) -> Tensor:
    multiplier = 10 ** decimals
    return torch.round(tensor * multiplier) / multiplier
    
def init_distributed(model: Module,
                     generator: TPModuleGenerator,
                     tp_group: Optional[dist.ProcessGroup] = None) -> Module:
    if tp_group is None:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        tp_group = dist.new_group(ranks=[k for k in range(world_size)], backend="nccl")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(rank)
    global TP_GROUP
    TP_GROUP = tp_group
        
    return generator(model, tp_group)