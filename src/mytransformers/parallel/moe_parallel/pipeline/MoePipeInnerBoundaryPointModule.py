from torch.distributed import ProcessGroup
from torch.nn import Module
import torch
from typing import Any
from mytransformers.parallel.pipeline_parallel.layers import PipeInnerBoundaryPointModule
from mytransformers.parallel.pipeline_parallel.layers.PipeModule import PipeRole
from mytransformers.parallel.pipeline_parallel.layers.strategies import InnerStrategyModule
from mytransformers.parallel.moe_parallel.pipeline.Scheduler import BaseScheduler

    
class MoePipeInnerBoundaryPointModule(PipeInnerBoundaryPointModule):
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
        super().__init__(role, module, current_group, comm_group, strategy)
        self.scheduler = scheduler
        self.thread_idx = 0
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Any:
        output = self.scheduler.transfer(op=self.strategy,
                                         op_info=self.thread_idx,
                                         output=self.module(*args, **kwargs),
                                         is_send=self.is_send,
                                         current_group=self.current_group,
                                         comm_group=self.comm_group)
        return output
    
    def reset(self) -> None:
        self.thread_idx = 0