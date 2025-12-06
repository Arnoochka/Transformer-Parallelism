from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Tuple, Dict
from .PipeGenerator import PipeGenerator
from mytransformers.parallel.pipeline_parallel.layers import (
    PipeBoundaryPointModule, PipeMainBoundaryPointModule,
    PipeFakeModule, PipeModule, PipeRole, StrategyModule, FakeModule)

class BoundaryPointModuleGenerator(PipeGenerator):
    """
    генератор, для модуля граничной точки внутри конвейерного паралллеизма. На нужных процессах создает модуль крайней точки, на остальных - фейковый
    
    Args:
        module (Module): модуль, от которого получается тензор для передачи текущего этапа
        current_group_info (Tuple[ProcessGroup, List[int]]): информация о текущей группе этапа
        next_group_info (Tuple[ProcessGroup, List[int]]): информация о следующей группе этапа
        strategy (StrategyModule): стратегия передачи данных
        strategy_kwargs (Dict): аргументы стратегии
    """
    def __new__(cls,
                module: Module,
                current_group_info: Tuple[ProcessGroup, List[int]],
                next_group_info: Tuple[ProcessGroup, List[int]],
                fake_module: FakeModule,
                strategy: StrategyModule,
                strategy_kwargs: Dict) -> PipeModule:
        rank = dist.get_rank()
        current_group, current_ranks = current_group_info
        next_group, next_ranks = next_group_info
        if rank in current_ranks:
            return PipeBoundaryPointModule(PipeRole.computeAndSend,
                                      module,
                                      current_group,
                                      next_group,
                                      strategy(**strategy_kwargs))
        elif rank in next_ranks:
            return PipeBoundaryPointModule(PipeRole.recv,
                                      fake_module,
                                      current_group,
                                      next_group,
                                      strategy(**strategy_kwargs))
        else:
            return PipeFakeModule(fake_module)
        
        
class MainBoundaryPointModuleGenerator(PipeGenerator):
    """
    генератор, для модуля крайнкй точки  конвейерного параллелизма.
    
    Args:
        module (Module): модуль, от которого получается тензор для передачи текущего этапа
        group (ProcessGroup): текущая группа этапа
        bcast_group (ProcessGroup): группа процессов, которой необходимо передать данные
        is_finished (bool): является крайняя точка конечной (если False, то она начальная)
        strategy (StrategyModule): стратегия передачи данных
        strategy_kwargs (Dict): аргументы стратегии
    """
    def __new__(cls,
                module: Module,
                current_group_info: ProcessGroup,
                bcast_group: ProcessGroup,
                is_finished: bool,
                strategy: StrategyModule,
                strategy_kwargs: Dict = {}) -> PipeModule:
        
        rank = dist.get_rank()
        current_group, current_ranks = current_group_info
        return PipeMainBoundaryPointModule(PipeRole.computeAndSend if is_finished else PipeRole.recv,
                                           module,
                                           current_group,
                                           bcast_group,
                                           strategy(**strategy_kwargs),
                                           is_finished)