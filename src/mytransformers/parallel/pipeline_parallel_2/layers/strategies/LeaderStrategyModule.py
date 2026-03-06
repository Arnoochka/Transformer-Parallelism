from torch.distributed import ProcessGroup
from .StrategyModule import InnerStrategyModule, COUNTER
import torch.distributed as dist
from torch import Tensor
import torch
from typing import Tuple, Dict, Optional, Any
from mytransformers import utils

class LeaderStrategyModule(InnerStrategyModule):
    """
    Стратегия передачи на основе выбора лидера:
        1. В каждой группе выбирается один процесс, который принимает или отправляет данные ("лидер")
        2. Происходит передача данных
        3. На принимающей группе делается операция broadcast для передачи данных всем процессам
        
    Args:
        send_leader (int): локальный ранг процесса в группе, который будет лидером-отправителем в comm_group
        recv_leader (int): локальный ранг процесса в группе, который будет лидером-получателем в comm_group
        bcast_leader (int): локальный ранг процесса в группе, который является лидером-получателем в comm_group
        и лидером-отправителем в current_group
    """
    def __init__(self,
                 send_leader: int = 0,
                 recv_leader: int = 1,
                 bcast_leader: int = 0):
        super().__init__()
        self.send_leader = send_leader
        self.recv_leader = recv_leader
        self.bcast_leader = bcast_leader
        
    def forward(self,
                output: Tensor,
                is_send: bool,
                current_group: ProcessGroup,
                comm_group: ProcessGroup) -> Tensor:
        """
        Args:
            output (Tensor): выход, который необходимо передать не следующий процесс
            is_send (bool): является ли процесс отправителем
            current_group (ProcessGroup): текущая группа процессов
            comm_group (ProcessGroup): группа процессов для коммуникации
            
        Returns:
            Tensor: выход, который необходимо быдло передать (output)
        """
        tag = next(COUNTER)
        torch.cuda.synchronize()
        if is_send:
            if dist.get_rank(comm_group) == self.send_leader:
                dst_rank = dist.get_global_rank(comm_group, self.recv_leader)
                dist.send(output.contiguous(), dst=dst_rank, group=comm_group, tag=tag)
        else:
            if dist.get_rank(comm_group) == self.recv_leader:
                src_rank = dist.get_global_rank(comm_group, self.send_leader)
                dist.recv(output.contiguous(), src=src_rank, group=comm_group, tag=tag) 
                
            src_rank = dist.get_global_rank(current_group, self.bcast_leader)
            dist.broadcast(output.contiguous(), src=src_rank, group=current_group)
        return output
        
class LeaderTupleStrategyModule(LeaderStrategyModule):
    """
    Стратегия передачи на основе выбора лидера:
        1. В каждой группе выбирается один процесс, который принимает или отправляет данные ("лидер")
        2. Происходит передача данных
        3. На принимающей группе делается операция broadcast для передачи данных всем процессам
        
    Args:
        send_leader (int): локальный ранг процесса в группе, который будет лидером-отправителем в comm_group
        recv_leader (int): локальный ранг процесса в группе, который будет лидером-получателем в comm_group
        bcast_leader (int): локальный ранг процесса в группе, который является лидером-получателем в comm_group
        и лидером-отправителем в current_group
    """
    def forward(self,
                output: Tuple[Any],
                is_send: bool,
                current_group: ProcessGroup,
                comm_group: ProcessGroup) -> Tuple[Any]:
        """
        Args:
            output (Tuple[Any]): выход, который необходимо передать не следующий процесс
            is_send (bool): является ли процесс отправителем
            current_group (ProcessGroup): текущая группа процессов
            comm_group (ProcessGroup): группа процессов для коммуникации
            
        Returns:
            Tuple[Any]: выход, который необходимо быдло передать (output)
        """
        new_output = []
        for out in output:
            if isinstance(out, Tensor):
                out = super().forward(out, is_send, current_group, comm_group)
            new_output.append(out)

        return tuple(new_output)
    

class LeaderStrategyDictModule(LeaderStrategyModule):
    """
    Стратегия передачи на основе выбора лидера:
        1. В каждой группе выбирается один процесс, который принимает или отправляет данные ("лидер")
        2. Происходит передача данных
        3. На принимающей группе делается операция broadcast для передачи данных всем процессам
        
    Args:
        send_leader (int): локальный ранг процесса в группе, который будет лидером-отправителем в comm_group
        recv_leader (int): локальный ранг процесса в группе, который будет лидером-получателем в comm_group
        bcast_leader (int): локальный ранг процесса в группе, который является лидером-получателем в comm_group
        и лидером-отправителем в current_group
    """
    def forward(self,
                output: Dict[str, Any],
                is_send: bool,
                current_group: ProcessGroup,
                comm_group: ProcessGroup) -> Dict[str, Any]:
        """
        Args:
            output (Dict[str, Any]): выход, который необходимо передать не следующий процесс
            is_send (bool): является ли процесс отправителем
            current_group (ProcessGroup): текущая группа процессов
            comm_group (ProcessGroup): группа процессов для коммуникации
            
        Returns:
            Dict[str, Any]: выход, который необходимо быдло передать (output)
        """
        new_output = {}
        for name, out in output.items():
            if isinstance(out, Tensor):
                out = super().forward(out, is_send, current_group, comm_group)
            new_output[name] = out
        return new_output