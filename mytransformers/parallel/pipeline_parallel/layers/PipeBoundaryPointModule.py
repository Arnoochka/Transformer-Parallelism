from torch.distributed import ProcessGroup
from .PipeModule import PipeModule, PipeRole
from .strategies import StrategyModule
from torch.nn import Module
from typing import Any, Callable
from mytransformers.utils import Logger
from .strategies import COUNTER


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
        self.callback = None
        
    def forward(self, *args, **kwargs) -> Any:
        output = self.module(*args, **kwargs)
        output = self.strategy(output,
                               self.is_send,
                               self.send_group,
                               self.recv_group)
        
        
        return output
    
    
class PipeMainBoundaryPointModule(PipeBoundaryPointModule):
    """
    крайняя точка (начальная или конечная) для конвейерного параллелизма.
    
    Задача: Broadcast данных на необходимые узлы в начале или конце конвейера
    
    Args:
        role (PipeRole): роль модуля (send, recv, computeAndSend)
        module (Module): модуль, от которого получается тензор для передачи
        send_group (ProcessGroup): группа, которая отправляет данные
        recv_group (ProcessGroup): группа, которая получает данные
        strategy (StrategyModule): стратегия передачи данных
        is_finished (bool): является ли точка конечной (если False, то она начальная)
    """
    
    def __init__(self,
                 role: PipeRole,
                 module: Module,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup,
                 strategy: StrategyModule,
                 is_finished: bool):
        super().__init__(role, module, send_group, recv_group, strategy)
        self.make_callback = None
        self.is_finished = is_finished
        
    def forward(self, *args, **kwargs) -> Any:
        output = self.module(*args, **kwargs)  
        output = self.strategy(output,
                               self.is_send,
                               self.send_group,
                               self.recv_group)
        self.make_callback(self.is_finished) 
        return output
        
    def set_callback(self, callback_func: Callable) -> None:
        self.make_callback = callback_func
    


        
                
    