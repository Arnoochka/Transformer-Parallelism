from torch.nn import Module
from mytransformers.parallel.ParallelModule import ParallelModule
from typing import Optional, Callable
from torch import Tensor

class PipeModule(ParallelModule):
    def __init__(self,
                 module: Module,
                 recv: bool = False,
                 recv_strategy: Optional[Callable] = None,
                 send: bool = False,
                 send_strategy: Optional[Callable] = None):
        super().__init__()
        self.module = module
        self.recv = recv
        self.recv_strategy = recv_strategy
        self.send = send
        self.send_strategy = send_strategy
    def forward(self, x: Tensor) -> Tensor:
        logits = self.module(x)
        if self.send: self.strategy(logits)
        return logits
        
    