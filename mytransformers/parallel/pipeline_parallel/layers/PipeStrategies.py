from torch.distributed import ProcessGroup
from .PipeModule import PipeModule, PipeRole
import torch.distributed as dist
from torch.nn import Module
from torch import Tensor
import torch

class PipeStrategy(PipeModule):
    def __init__(self,
                 role: PipeRole,
                 module: Module,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup,
                 tensor_dim: int = 3):
        super().__init__(role, module)
        """
        args:
            role: send or receive
            send_group: group with send role
            recv_group: group with recieve role
            tensor_dim: dim of tensor that is transfrered
        return: transfered tensor
        """
        self.send_group = send_group
        self.recv_group = recv_group
        self.tensor_dim = tensor_dim
        
    def forward(self, *args, **kwargs) -> Tensor:
        x = super().forward(*args, **kwargs)
        return x
    
    
class PipeLeaderStrategy(PipeStrategy):
    def __init__(self,
                 role: PipeRole,
                 module: Module,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup,
                 tensor_dim: int = 3,
                 leader_rank: int = 0):
        super().__init__(role, module, send_group, recv_group, tensor_dim)
        self.leader_rank = leader_rank

    def forward(self, *args, **kwargs) -> Tensor:
        x = super().forward(*args, **kwargs)

        if self.role in (PipeRole.send, PipeRole.computeAndSend):
            if dist.get_rank(group=self.send_group) == self.leader_rank:
                dst_rank = dist.get_global_rank(self.recv_group, self.leader_rank)
                shape_tensor = torch.tensor(list(x.size()), dtype=torch.long, device=x.device)
                dist.send(shape_tensor, dst_rank, tag=0)
                dist.send(x, dst_rank, tag=1)

        else:
            if dist.get_rank(group=self.recv_group) == self.leader_rank:
                src_rank = dist.get_global_rank(self.send_group, self.leader_rank)

                shape_tensor = torch.zeros(self.tensor_dim, dtype=torch.long, device=x.device)
                dist.recv(shape_tensor, src=src_rank, tag=0)
                dist.broadcast(shape_tensor, src=self.leader_rank, group=self.recv_group)

                size_list = shape_tensor.tolist()
                x_recv = torch.zeros(size_list, dtype=x.dtype, device=x.device)
                dist.recv(x_recv, src=src_rank, tag=1)
                dist.broadcast(x_recv, src=self.leader_rank, group=self.recv_group)
                x = x_recv
            else:
                shape_tensor = torch.zeros(self.tensor_dim, dtype=torch.long, device=x.device)
                dist.broadcast(shape_tensor, src=self.leader_rank, group=self.recv_group)
                size_list = shape_tensor.tolist()
                x_local = torch.zeros(size_list, dtype=x.dtype, device=x.device)
                dist.broadcast(x_local, src=self.leader_rank, group=self.recv_group)
                x = x_local

        return x

    
class PipeBroadcastLeaderStrategy(PipeLeaderStrategy):
    def forward(self, *args, **kwargs) -> Tensor:
        x = super().forward(*args, **kwargs)
        src_rank = dist.get_global_rank(self.send_group, self.leader_rank)

        if self.role in (PipeRole.send, PipeRole.computeAndSend) and dist.get_rank(group=self.send_group) == self.leader_rank:
            shape_tensor = torch.tensor(list(x.size()), dtype=torch.long, device=x.device)
            dist.broadcast(shape_tensor, src=src_rank, group=self.recv_group)
            dist.broadcast(x, src=src_rank, group=self.recv_group)
        else:
            shape_tensor = torch.zeros(self.tensor_dim, dtype=torch.long, device=x.device)
            dist.broadcast(shape_tensor, src=src_rank, group=self.recv_group)
            size_list = shape_tensor.tolist()
            x = torch.zeros(size_list, dtype=x.dtype, device=x.device)
            dist.broadcast(x, src=src_rank, group=self.recv_group)

        return x

        
                
    