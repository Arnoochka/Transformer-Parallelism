from torch.distributed import ProcessGroup
from .StrategyModule import StrategyModule
import torch.distributed as dist
from torch import Tensor
import torch
from typing import Dict, Optional, Any, Tuple
from mytransformers.utils import Logger

class FinalStrategyModule(StrategyModule):
    """
    класс финальной стратегии Pipeline
    
    Args:
        send_rank (int): ранк, с которого отправляются данные
    """
    def __init__(self, 
                 send_rank: int):
        super().__init__()
        self.send_rank = send_rank
    def forward(self,
                output: Tensor,
                is_send: bool,
                send_group: ProcessGroup,
                recv_group: ProcessGroup,
                comm_group: ProcessGroup) -> Tensor:
        torch.cuda.synchronize()
        dist.broadcast(output.contiguous(), src=self.send_rank, group=comm_group)
        return output
    
class FinalStrategyTupleModule(FinalStrategyModule):

    def forward(self,
                output: Tuple[Any],
                is_send: bool,
                send_group: ProcessGroup,
                recv_group: ProcessGroup,
                comm_group: ProcessGroup) -> Tensor:
        new_output = {}
        for name, out in output:
            if isinstance(out, Tensor):
                out = super().forward(out, is_send, send_group, recv_group, comm_group)
            new_output[name] = out
        return new_output
    
class FinalStrategyDictModule(FinalStrategyModule):

    def forward(self,
                output: Dict[str, Any],
                is_send: bool,
                send_group: ProcessGroup,
                recv_group: ProcessGroup,
                comm_group: ProcessGroup) -> Tensor:
        new_output = {}
        for name, out in output.items():
            if isinstance(out, Tensor):
                out = super().forward(out, is_send, send_group, recv_group, comm_group)
            new_output[name] = out
        return new_output