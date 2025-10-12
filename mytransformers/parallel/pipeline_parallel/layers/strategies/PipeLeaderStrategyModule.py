from torch.distributed import ProcessGroup
from mytransformers.parallel.pipeline_parallel.layers.PipeModule import  PipeRole
from .PipeStrategyModule import PipeStrategyModule
import torch.distributed as dist
from torch.nn import Module
from torch import Tensor
import torch
from typing import Tuple

class PipeLeaderStrategyModule(PipeStrategyModule):
    def __init__(self,
                 role: PipeRole,
                 module: Module,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup,
                 tensor_dim: int = 3,
                 leader_rank: int = 0):
        super().__init__(role, module, send_group, recv_group, tensor_dim)
        
        self.send_leader_rank = dist.get_global_rank(self.send_group, leader_rank)
        self.recv_leader_rank = dist.get_global_rank(self.recv_group, leader_rank)
    
    def transfer_by_strategy(self, output: Tensor, type_transfer="tensor") -> Tensor:
        torch.cuda.synchronize(device=output.device)
        if self.is_send:
            if dist.get_rank() == self.send_leader_rank:
                shape_tensor = torch.tensor(list(output.size()), dtype=torch.long, device=output.device)
                dist.send(shape_tensor, self.recv_leader_rank, tag=0)
                dist.send(output, self.recv_leader_rank, tag=1)
        else:
            if dist.get_rank() == self.recv_leader_rank:
                shape_tensor = torch.zeros(self.tensor_dim, dtype=torch.long, device=output.device)
                dist.recv(shape_tensor, src=self.send_leader_rank, tag=0)
                dist.broadcast(shape_tensor, src=self.recv_leader_rank, group=self.recv_group)
                size_list = shape_tensor.tolist()
                output_recv = torch.zeros(size_list, dtype=output.dtype, device=output.device)
                dist.recv(output_recv, src=self.send_leader_rank, tag=1)
                dist.broadcast(output_recv, src=self.recv_leader_rank, group=self.recv_group)
                output = output_recv
            else:
                shape_tensor = torch.zeros(self.tensor_dim, dtype=torch.long, device=output.device)
                dist.broadcast(shape_tensor, src=self.recv_leader_rank, group=self.recv_group)
                size_list = shape_tensor.tolist()
                output_local = torch.zeros(size_list, dtype=output.dtype, device=output.device)
                dist.broadcast(output_local, src=self.recv_leader_rank, group=self.recv_group)
                output = output_local
        torch.cuda.synchronize(device=output.device)  
        return output
        
class PipeLeaderTupleStrategyModule(PipeLeaderStrategyModule):
    def transfer_by_strategy(self, output: Tuple) -> Tuple:
        parent_strategy = super().transfer_by_strategy
        output = [parent_strategy(out, "tuple") for out in output]
        return tuple(output)
