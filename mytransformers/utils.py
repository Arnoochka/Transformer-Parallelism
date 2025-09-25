import os
from typing import Optional
import torch
from torch import Tensor
from torch.nn import Module
from mytransformers.parallel.tensor_parallel import TPModuleGenerator
import torch.distributed as dist
from typing import Any, List
from enum import Enum

GROUP = None

class MemoryUnits(Enum):
    GB = 1024**3
    MB = 1024**2
    KB = 1024


def logger(log: Any, rank: int) -> None:
    if rank == 0:
        if not isinstance(log, str):
            log = f"{log}"
        print(log)
        
def get_prompts(filename: str) -> List[str]:
    with open(filename, 'r') as file:
        promts = [line for line in file]
        return promts

def torch_round(tensor: Tensor, decimals: int) -> Tensor:
    multiplier = 10 ** decimals
    return torch.round(tensor * multiplier) / multiplier

def get_model_size(model: Module, unit = MemoryUnits.GB):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return total_size / unit.value
    
def init_distributed() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    group = dist.new_group(ranks=[k for k in range(world_size)], backend="nccl")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(rank)
    global GROUP
    GROUP = group
    return group