from torch.distributed import ProcessGroup
from .PipeModule import PipeModule, PipeRole
from .strategies import StrategyModule
from torch.nn import Module
from typing import Any
    


class PipeBoundaryPointModule(PipeModule):
    """
    крайняя точка (начальная или конечная) внутри конвейера.
    
    Задача: передача данных с одного GPU на другой для прохода следующего этапа
    
    Args:
        role (PipeRole): роль модуля (send, recv, computeAndSend)
        module (Module): модуль, от которого получается тензор для передачи
        current_group (ProcessGroup): текущая группа процессов
        comm_group (ProcessGroup): группа процессов для коммуникации
        strategy (StrategyModule): стратегия передачи данных
    """
    def __init__(self,
                 role: PipeRole,
                 module: Module,
                 current_group: ProcessGroup,
                 comm_group: ProcessGroup,
                 strategy: StrategyModule):
        super().__init__(role, module)
        is_send = (role == PipeRole.computeAndSend)
        is_recv = (role  == PipeRole.recv)
        if not (is_send or is_recv):
            raise AttributeError(f"role: {role}, strategy type is not selected")   
        
        self.is_send = is_send
        self.current_group = current_group
        self.comm_group = comm_group
        self.strategy = strategy
        
    def forward(self, *args, **kwargs) -> Any:
        
        output = self.strategy(self.module(*args, **kwargs),
                               self.is_send,
                               self.current_group,
                               self.comm_group)
        return output