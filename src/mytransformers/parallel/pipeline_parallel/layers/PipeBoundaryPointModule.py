from torch.distributed import ProcessGroup
from .PipeModule import PipeModule, PipeRole
from .strategies import StrategyModule
from torch.nn import Module
from typing import Any, Callable
from threading import Lock


class Mutex:
    lock = Lock()
    
    @staticmethod
    def callback(is_completed: bool) -> None:
        if is_completed:
            Mutex.lock.release()
        else:
            Mutex.lock.acquire()
            
    


class PipeBoundaryPointModule(PipeModule):
    """
    крайняя точка (начальная или конечная) внутри конвейера.
    
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
        super().__init__(role, module)
        is_send = (role == PipeRole.computeAndSend)
        is_recv = (role  == PipeRole.recv)
        if not (is_send or is_recv):
            raise AttributeError(f"role: {role}, strategy type is not selected")   
        
        self.is_send = is_send
        self.send_group = send_group
        self.recv_group = recv_group
        self.strategy = strategy
        self.callback: Callable[[bool], None] = None
        
    def forward(self, *args, **kwargs) -> Any:
        output = self.module(*args, **kwargs)
        self.callback(False)
        output = self.strategy(output,
                               self.is_send,
                               self.send_group,
                               self.recv_group)
        self.callback(True)
        return output