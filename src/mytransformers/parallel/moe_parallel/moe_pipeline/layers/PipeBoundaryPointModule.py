from torch.distributed import ProcessGroup
from .PipeModule import PipeModule, PipeRole
from .strategies import InnerStrategyModule
from mytransformers.parallel.moe_parallel.moe_pipeline.pipeline.Scheduler import BaseScheduler
from torch.nn import Module
from typing import Any
    
class PipeInnerBoundaryPointModule(PipeModule):
    """
    крайняя точка (начальная или конечная) внутри конвейера.
    
    Задача: передача данных с одного GPU на другой для прохода следующего этапа
    
    Args:
        role (PipeRole): роль модуля (send, recv, computeAndSend)
        module (Module): модуль, от которого получается тензор для передачи
        current_group (ProcessGroup): текущая группа процессов
        comm_group (ProcessGroup): группа процессов для коммуникации
        strategy (InnerStrategyModule): стратегия передачи данных
        scheduler (BaseScheduler): расписание для коллективных операций
    """
    def __init__(self,
                 role: PipeRole,
                 module: Module,
                 current_group: ProcessGroup,
                 comm_group: ProcessGroup,
                 strategy: InnerStrategyModule,
                 scheduler: BaseScheduler):
        super().__init__(role, module)
        is_send = (role == PipeRole.computeAndSend)
        is_recv = (role  == PipeRole.recv)
        if not (is_send or is_recv):
            raise AttributeError(f"role: {role}, strategy type is not selected")   
        
        self.is_send = is_send
        self.current_group = current_group
        self.comm_group = comm_group
        self.strategy = strategy
        self.scheduler = scheduler
        self.thread_idx = 0
        
    def forward(self, *args, **kwargs) -> Any:
        output = self.module(*args, **kwargs)
        output = self.scheduler.transfer(op=self.strategy,
                                         op_info=self.thread_idx,
                                         output=output,
                                         is_send=self.is_send,
                                         current_group=self.current_group,
                                         comm_group=self.comm_group)
        self.thread_idx += 1
        return output
    
    def reset(self) -> None:
        self.thread_idx = 0