from torch.distributed import ProcessGroup
from mytransformers.parallel.pipeline_parallel.layers.PipeModule import  PipeRole
from .PipeStrategyModule import PipeStrategyModule
import torch.distributed as dist
from torch.nn import Module
from torch import Tensor
import torch
        


class PipeLeaderStrategyModule(PipeStrategyModule):
    def __init__(self,
                 role: PipeRole,
                 module: Module,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup,
                 tensor_dim: int = 3,
                 leader_rank: int = 0):
        super().__init__(role, module, send_group, recv_group, tensor_dim)
        self.leader_rank = leader_rank
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Tensor:
        x: Tensor = self.module(*args, **kwargs)

        if self.is_send:
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

@torch.no_grad()   
class PipeBroadcastLeaderStrategyModule(PipeLeaderStrategyModule):
    def forward(self, *args, **kwargs) -> Tensor:
        x: Tensor = self.module(*args, **kwargs)
        src_rank = dist.get_global_rank(self.send_group, self.leader_rank)

        shape_tensor = torch.zeros(self.tensor_dim, dtype=torch.long, device=x.device)
        dist.broadcast(shape_tensor, src=src_rank)
        size_list = shape_tensor.tolist()
        if not self.is_send:
            x = torch.zeros(size_list, dtype=x.dtype, device=x.device)
        dist.broadcast(x, src=src_rank)

        return x
