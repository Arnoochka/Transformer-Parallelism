from torch.distributed import ProcessGroup
from .StrategyModule import StrategyModule, COUNTER
import torch.distributed as dist
from torch import Tensor
from typing import Tuple, Dict, Optional
from mytransformers.benchmark import get_global_tracker

class LeaderStrategyModule(StrategyModule):
    """
    Стратегия передачи на основе выбора лидера:
        1. В каждой группе выбирается один процесс, который принимает или отправляет данные ("лидер")
        2. Происходит передача данных
        3. На принимающей группе делается операция broadcast для передачи данных всем процессам
        
    Args:
        leader_rank (int): локальный ранг процесса в группе, который будет "лидером"
    """
    def __init__(self, leader_rank: int = 0):
        super().__init__()
        self.leader_rank = leader_rank
        
    def forward(self,
                output: Tensor,
                is_send: bool,
                send_group: ProcessGroup,
                recv_group: ProcessGroup) -> Tensor:
        """
        Args:
            output (Tensor): выход, который необходимо передать не следующий процесс
            is_send (bool): является ли процесс отправителем
            send_group (ProcessGroup): группа, процессов, которая отправляет данные
            recv_group (ProcessGroup): группа процессов, которая принимает данные
            
        Returns:
            Tensor: выход, который необходимо быдло передать (output)
        """
        tag = next(COUNTER)
        send_leader_rank = dist.get_global_rank(send_group, self.leader_rank)
        recv_leader_rank = dist.get_global_rank(recv_group, self.leader_rank)
        if is_send:
            if dist.get_rank() == send_leader_rank:
                dist.send(output, recv_leader_rank, tag=tag)
        else:
            if dist.get_rank() == recv_leader_rank:
                dist.recv(output, src=send_leader_rank, tag=tag)
            dist.broadcast(output, src=recv_leader_rank, group=recv_group)
        return output
        
class LeaderTupleStrategyModule(LeaderStrategyModule):
    """
    Стратегия передачи на основе выбора лидера, однако вместо тензора передается кортеж
        1. В каждой группе выбирается один процесс, который принимает или отправляет данные ("лидер")
        2. Происходит передача данных
        3. На принимающей группе делается операция broadcast для передачи данных всем процессам
        
    Args:
        leader_rank (int): локальный ранг процесса в группе, который будет "лидером"
    """
    def forward(self,
                output: Tuple[Optional[Tensor]],
                is_send: bool,
                send_group: ProcessGroup,
                recv_group: ProcessGroup) -> Tuple[Optional[Tensor]]:
        """
        Args:
            output (Tuple[Optional[Tensor]]): выход, который необходимо передать не следующий процесс
            is_send (bool): является ли процесс отправителем
            send_group (ProcessGroup): группа, процессов, которая отправляет данные
            recv_group (ProcessGroup): группа процессов, которая принимает данные
            
        Returns:
            Tuple[Optional[Tensor]]: выход, который необходимо было передать (output)
        """
        TRACKER = get_global_tracker()
        TRACKER.snapshot("START inner strategy")
        new_output = []
        for out in output:
            if out is not None:
                out = super().forward(out,is_send,send_group,recv_group)
            new_output.append(out)
        TRACKER.snapshot("END inner strategy")
        return tuple(new_output)
    

class LeaderStrategyDictModule(LeaderStrategyModule):
    """
    Стратегия передачи на основе выбора лидера, однако вместо тензора передается словарь
        1. В каждой группе выбирается один процесс, который принимает или отправляет данные ("лидер")
        2. Происходит передача данных
        3. На принимающей группе делается операция broadcast для передачи данных всем процессам
        
    Args:
        leader_rank (int): локальный ранг процесса в группе, который будет "лидером"
    """
    def forward(self, 
                output: Dict[str, Optional[Tensor]],
                is_send: bool,
                send_group: ProcessGroup,
                recv_group: ProcessGroup) -> Dict[str, Optional[Tensor]]:
        """
        Args:
            output (Dict[str, Optional[Tensor]]): выход, который необходимо передать не следующий процесс
            is_send (bool): является ли процесс отправителем
            send_group (ProcessGroup): группа, процессов, которая отправляет данные
            recv_group (ProcessGroup): группа процессов, которая принимает данные
            
        Returns:
            Dict[str, Optional[Tensor]]: выход, который необходимо было передать (output)
        """
        new_output = {}
        for name, out in output.items():
            if out is not None:
                out = super().forward(out,is_send,send_group,recv_group)
            new_output[name] = out
        return new_output