from torch.distributed import ProcessGroup
from mytransformers.parallel.pipeline_parallel.layers.PipeModule import PipeModule, PipeRole
from .strategies import StrategyModule
from torch.nn import Module
from typing import Any

class PipeStrategyModule(PipeModule):
    """
    Стратегический модуль для конвейерного параллелизма.
    
    Задача: передача данных с одного GPU на другой для прохода следующего этапа
    
    Args:
        role (PipeRole): роль модуля (send, recv, computeAndSend)
        module (Module): модуль, от которого получается тензор для передачи
        send_group (ProcessGroup): группа, которая отправляет данные
        recv_group (ProcessGroup): группа, которая получает данные
        strategy (StrategyModule): стратегия передачи данных
    """
    def __init__(self,
                 role: PipeRole,
                 module: Module,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup,
                 strategy: StrategyModule):
        super().__init__(role)
        is_send = (role == PipeRole.computeAndSend)
        is_recv = (role  == PipeRole.recv)
        if not (is_send or is_recv):
            raise AttributeError(f"role: {role}, strategy type is not selected")
        
        self.is_send = is_send
        self.send_group = send_group
        self.recv_group = recv_group
        self.module = module
        self.strategy = strategy
        
    def forward(self, *args, **kwargs) -> Any:
        output = self.module(*args, **kwargs)
        output = self.strategy(output,
                               self.is_send,
                               self.send_group,
                               self.recv_group)
        self.complete()
        return output
    
    def complete(self) -> None:
        self.strategy.wait()
    


        
                
    