from torch.distributed import ProcessGroup
from .PipeModule import PipeModule, PipeRole
from .fake_modules.FakeArgsSetter import FakeArgsSetter
from .strategies import StrategyModule
from torch.nn import Module
from typing import Any, Callable
from enum import Enum

class BoundaryPointRole(Enum):
    start = 0
    end = 1

class PipeBoundaryPointModule(PipeModule):
    """
    крайняя точка (начальная или конечная) для конвейерного параллелизма.
    
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
                 strategy: StrategyModule,
                 fake_args_setter: FakeArgsSetter):
        super().__init__(role)
        is_send = (role == PipeRole.computeAndSend)
        is_recv = (role  == PipeRole.recv)
        if not (is_send or is_recv):
            raise AttributeError(f"role: {role}, strategy type is not selected")
        
        if is_recv:
            self.point_role = BoundaryPointRole.start
        else:
            self.point_role = BoundaryPointRole.end    
        
        self.is_send = is_send
        self.send_group = send_group
        self.recv_group = recv_group
        self.module = module
        self.strategy = strategy
        self.set_fake_args = fake_args_setter
        
    def forward(self, *args, **kwargs) -> Any:
        output = self.module(*args, **kwargs)
        output = self.strategy(output,
                               self.is_send,
                               self.send_group,
                               self.recv_group)
        if self.point_role == BoundaryPointRole.end:
            self.set_fake_args(output)
            
        return output
    


        
                
    