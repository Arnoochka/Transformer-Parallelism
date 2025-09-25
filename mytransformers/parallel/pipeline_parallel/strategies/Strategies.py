from torch.distributed import ProcessGroup
from mytransformers.parallel.ParallelModule import ParallelModule
import torch.distributed as dist
from typing import Tuple
from torch import Tensor
from enum import Enum
import torch

class CommRole(Enum):
    send = 0
    recv = 1
    none = 2

class Strategy(ParallelModule):
    def __init__(self,
                 comm_role: CommRole,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup,
                 tensor_dim: int = 3):
        super().__init__()
        """
        args:
            comm_role: send or receive
            send_group: group with send role
            recv_group: group with recieve role
            tensor_size: size of tensor that is transfrered
        return: transfered tensor
        """
        self.comm_role = comm_role
        self.send_group = send_group
        self.recv_group = recv_group
        self.tensor_dim = tensor_dim
        
    def forward(self, x: Tensor) -> Tensor:
        if self.comm_role == CommRole.none:
            raise AttributeError(f"comm role is CommRole.none")
        return x
    
    
class LeaderStrategy(Strategy):
    def __init__(self,
                 comm_role: CommRole,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup,
                 tensor_size: Tuple[int],
                 leader_rank: int = 0):
        """
        args:
            comm_role: send or receive
            send_group: group with send role
            recv_group: group with recieve role
            tensor_size: size of tensor that is transfrered
            leader rank: local rank in comm group that send or receive tensor
        """
        super().__init__(comm_role,
                         send_group,
                         recv_group,
                         tensor_size)
        self.leader_rank = leader_rank
        self.tag = 0
        
    def forward(self, x: Tensor) -> Tensor:
        super().forward(x)
        if self.comm_role == CommRole.send:
            if dist.get_rank(self.send_group) == self.leader_rank:
                dst_rank = dist.get_global_rank(self.recv_group, self.leader_rank)
                dist.send(torch.tensor(x.size()), dst_rank)
                dist.send(x, dst_rank, tag=self.tag)
        else:
            if dist.get_rank(self.recv_group) == self.leader_rank:
                src_rank = dist.get_global_rank(self.send_group, self.leader_rank)
                tensor_size = torch.zeros(self.tensor_dim)
                dist.recv(tensor_size, src_rank)
                dist.broadcast(tensor_size)
                x = torch.zeros(tensor_size)
                dist.recv(x, src_rank, tag=self.tag)
                dist.broadcast(x, self.leader_rank, group=self.recv_group)
            else:
                tensor_size = torch.zeros(self.tensor_dim)
                dist.broadcast(tensor_size, self.leader_rank, group=self.recv_group)
                x = torch.zeros(tensor_size)
                dist.broadcast(x, self.leader_rank, group=self.recv_group)
                
        self.tag += 1
        return x
                
    