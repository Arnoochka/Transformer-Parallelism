from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Tuple
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import (
    PipeBoundaryPointModule, PipeFakeModule, PipeModule,
    PipeRole, StrategyModule, FakeModule)

class BoundaryPointModuleGenerator(ParallelModuleGenerator):
    """
    генератор, для модуля граничной точки внутри конвейерного паралллеизма. На нужных процессах создает модуль крайней точки, на остальных - фейковый
    
    Args:
        module (Module): модуль, от которого получается тензор для передачи текущего этапа
        current_group_info (Tuple[ProcessGroup, List[int]]): информация о текущей группе этапа
        next_group_info (Tuple[ProcessGroup, List[int]]): информация о следующей группе этапа
        comm_group (ProcessGroup): группа для передачи данных
        strategy (StrategyModule): стратегия передачи данных
    """
    def __new__(cls,
                module: Module,
                current_group_info: Tuple[ProcessGroup, List[int]],
                next_group_info: Tuple[ProcessGroup, List[int]],
                comm_group: ProcessGroup,
                fake_module: FakeModule,
                strategy: StrategyModule) -> PipeModule:
        rank = dist.get_rank()
        current_group, current_ranks = current_group_info
        next_group, next_ranks = next_group_info
        if (rank in current_ranks) or (rank in next_ranks):
            role = PipeRole.computeAndSend if rank in current_ranks else PipeRole.recv
            actual_module = module if role == PipeRole.computeAndSend else fake_module
            actual_group = current_group if role == PipeRole.computeAndSend else next_group
            return PipeBoundaryPointModule(role,
                                           actual_module,
                                           actual_group,
                                           comm_group,
                                           strategy)
        else:
            return PipeFakeModule(fake_module)