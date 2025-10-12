import os
from typing import List
import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
from typing import Any, List
from enum import Enum
from torch.distributed import ProcessGroup
from torch.distributed import get_rank

GROUP: ProcessGroup = None
RANKS: List = None
BACKEND: str = None

class MemoryUnits(Enum):
    GB = 1024**3
    MB = 1024**2
    KB = 1024


class Logger:
    @staticmethod
    def log_main_device(log: Any, rank: int) -> None:
        if rank == 0:
            if not isinstance(log, str):
                log = f"{log}"
            print(log)
            
    @staticmethod
    def log_all_device(log: Any) -> None:
        if not isinstance(log, str):
            log = f"{log}"
        print(f"---device:{get_rank()}---:\n{log}")
            
            
            
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
    
def init_distributed(backend: str = 'nccl') -> None:
    global GROUP, BACKEND, RANKS
    BACKEND = backend
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=BACKEND, world_size=world_size, rank=rank)
    RANKS = [k for k in range(world_size)]
    GROUP = dist.new_group(ranks=RANKS, backend=BACKEND)
    torch.manual_seed(0)  

def init_distributed_cuda() -> None:
    init_distributed('nccl')
    rank = int(os.environ["RANK"])
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(rank)

def create_group(ranks: List[int]) -> dist.ProcessGroup:
    group = dist.new_group(ranks=ranks, backend=BACKEND)
    return group

