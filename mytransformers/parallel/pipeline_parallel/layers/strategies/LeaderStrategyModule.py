from torch.distributed import ProcessGroup
from .StrategyModule import StrategyModule, GLOBAL_COUNTER
import torch.distributed as dist
from torch.distributed import Work
from torch import Tensor
from typing import Tuple, List
from mytransformers.utils import Logger

class LeaderStrategyModule(StrategyModule):
    def __init__(self, leader_rank: int = 0):
        super().__init__()
        self.leader_rank = leader_rank
        
    def forward(self,
                output: Tensor,
                is_send: bool,
                send_group: ProcessGroup,
                recv_group: ProcessGroup) -> Tensor:
        self.tag = next(GLOBAL_COUNTER)
        send_leader_rank = dist.get_global_rank(send_group, self.leader_rank)
        recv_leader_rank = dist.get_global_rank(recv_group, self.leader_rank)
        if is_send:
            if dist.get_rank() == send_leader_rank:
                worker = dist.isend(output, recv_leader_rank, tag=0)
            else: worker = None
            self.workers.append(worker)
        else:
            if dist.get_rank() == recv_leader_rank:
                worker = dist.irecv(output, src=send_leader_rank, tag=0)
            else: worker = None
            self.workers.append(worker)
            worker = dist.broadcast(output, src=recv_leader_rank, group=recv_group, async_op=True)
            self.workers.append(worker)
        return output
        
class LeaderTupleStrategyModule(LeaderStrategyModule):
    def __init__(self, leader_rank: int = 0):
        super().__init__(leader_rank)
    def forward(self,
                output: Tuple[Tensor],
                is_send: bool,
                send_group: ProcessGroup,
                recv_group: ProcessGroup) -> Tuple[Tuple[Tensor], List[Work]]:
        new_output = []
        for out in output:
            out = super().forward(out,is_send,send_group,recv_group)
            new_output.append(out)
        return tuple(new_output)
