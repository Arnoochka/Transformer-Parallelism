from torch.distributed import ProcessGroup
from .StrategyModule import StrategyModule, COUNTER
import torch.distributed as dist
from torch import Tensor
import torch
from typing import Dict, Optional, Any

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
                comm_group: Optional[ProcessGroup] = None) -> Tensor:
        """
        Args:
            output (Tensor): выход, который необходимо передать не следующий процесс
            comm_group (optional[ProcessGroup]): группа процессов для коммуникации (если установлена None, то берется глобальная группа)
            
        Returns:
            Tensor: выход, который необходимо быдло передать (output)
        """
        tag = next(COUNTER)
        torch.cuda.synchronize()
        dist.broadcast(output.contiguous(), src=self.send_rank, group=comm_group)
        return output
    
class FinalStrategyDictModule(FinalStrategyModule):
    """
    класс финальной стратегии Pipeline
    
    Args:
        send_rank (int): ранк, с которого отправляются данные
    """
    def forward(self,
                output: Dict[str, Any],
                comm_group: Optional[ProcessGroup] = None) -> Dict[str, Any]:
        """
        Args:
            output (Dict[str, Any]): выход, который необходимо передать не следующий процесс
            comm_group (optional[ProcessGroup]): группа процессов для коммуникации (если установлена None, то берется глобальная группа)
            
        Returns:
            Dict[str, Any]: выход, который необходимо быдло передать (output)
        """
        new_output = {}
        for name, out in output.items():
            if isinstance(out, Tensor):
                out = super().forward(out, comm_group)
            new_output[name] = out
        return new_output