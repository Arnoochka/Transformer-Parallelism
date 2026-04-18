from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Tuple
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.moe_parallel.moe_pipeline.layers import (
    PipeInnerBoundaryPointModule, PipeFakeModule, PipeModule,
    PipeRole, InnerStrategyModule, FakeModule)
from mytransformers.parallel.moe_parallel.moe_pipeline.pipeline import BaseScheduler

class InnerBoundaryPointModuleGenerator(ParallelModuleGenerator):
    """
    генератор, для модуля граничной точки внутри конвейерного паралллеизма. На нужных процессах создает модуль крайней точки, на остальных - фейковый
    
    Args:
        module (Module): модуль, от которого получается тензор для передачи текущего этапа
        current_group_info (Tuple[ProcessGroup, List[int]]): информация о текущей группе этапа
        next_group_info (Tuple[ProcessGroup, List[int]]): информация о следующей группе этапа
        comm_group (ProcessGroup): группа для передачи данных
        strategy (InnerStrategyModule): стратегия передачи данных
        scheduler (BaseScheduler): расписание для коллективных операций
    """
    def __new__(cls,
                module: Module,
                current_group_info: Tuple[ProcessGroup, List[int]],
                next_group_info: Tuple[ProcessGroup, List[int]],
                comm_group: ProcessGroup,
                fake_module: FakeModule,
                strategy: InnerStrategyModule,
                scheduler: BaseScheduler) -> PipeModule:
        rank = dist.get_rank()
        current_group, current_ranks = current_group_info
        next_group, next_ranks = next_group_info
        if (rank in current_ranks) or (rank in next_ranks):
            role = PipeRole.computeAndSend if rank in current_ranks else PipeRole.recv
            actual_module = module if role == PipeRole.computeAndSend else fake_module
            actual_group = current_group if role == PipeRole.computeAndSend else next_group
            return PipeInnerBoundaryPointModule(role,
                                                actual_module,
                                                actual_group,
                                                comm_group,
                                                strategy,
                                                scheduler)
        else:
            return PipeFakeModule(fake_module)