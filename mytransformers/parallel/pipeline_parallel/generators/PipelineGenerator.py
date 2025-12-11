from mytransformers.parallel.pipeline_parallel.layers import (FakeModule, StrategyModule)
from .BoundaryPointModuleGenerator import BoundaryPointModuleGenerator
from .ComputeModuleGenerator import ComputeModuleGenerator
from typing import List, Tuple, Callable
from torch.nn import Module, ModuleList, ModuleDict
from torch.distributed import ProcessGroup
import torch.distributed as dist
from mytransformers.parallel.pipeline_parallel.Pipeline import Pipeline
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
    
    
class PipelineGenerator(ParallelModuleGenerator):
    """
    Генератор конвейерного параллелизма
    
    Args:
        model (Module): исходная модель
        modules (ModuleList): модули, которые подменили
        fake_args (Callable): генератор аргкментов для "фейковых" слоев
    """
    def __new__(cls,
                model: Module,
                modules: ModuleList,
                final_strategy: StrategyModule,
                last_group_info: Tuple[ProcessGroup, Tuple[int]],
                final_group: Tuple[ProcessGroup, Tuple[int]],
                fake_args: Callable) -> Module:
        rank = dist.get_rank()
        last_group, last_ranks = last_group_info
        final_strategy_args = (
            True if rank in last_ranks else False,
            last_group,
            final_group)
        
        pipeline = Pipeline(model.forward,
                            modules,
                            final_strategy,
                            final_strategy_args,
                            fake_args)
        
        model.forward = pipeline.forward
        return model
    
    @staticmethod
    def get_stage(modules: List[Tuple[str, Module, FakeModule]],
                  inner_boundary_points: List[int],
                  groups_info: List[Tuple[ProcessGroup, List[int]]],
                  inner_strategies: List[StrategyModule]) -> ModuleDict:
        """
        Генерирует фактическую стадию

        Args:
            model (Module): исходная модель
            modules (List[Tuple[str, Module, FakeModule]]): модули на подмену, (имя, оригинальный модуль, фейковый модуль)
            inner_boundary_points (List[int]): внутренние точки стадий
            groups_info (List[Tuple[ProcessGroup, List[int]]]): информация о группах стадий (len(groups_info) == len(inner_boundary_points) + 1)
            inner_strategies (List[StrategyModule]): стратегии для передачи данных между стадиями
            bcast_groups (Tuple[ProcessGroup, ProcessGroup]): каждому процессу внутри группы рассылаются данные на начальном или конечном слое
            bcast_strategies (Tuple[StrategyModule, StrategyModule]): стратегии для рассылки на начальном или конечном слое
            fake_args (Callable): генератор аргкментов для "фейковых" слоев
            num_microbatches (int): число микробатчей
        """
        
        stage = ModuleDict()
        inner_point_idx = 0
        for idx, (name, module, fake_module) in enumerate(modules):
            if idx in inner_boundary_points:
                pipe_module = BoundaryPointModuleGenerator(module,
                                                           groups_info[inner_point_idx],
                                                           groups_info[inner_point_idx + 1],
                                                           fake_module,
                                                           inner_strategies[inner_point_idx])
                inner_point_idx += 1
            else:
                pipe_module = ComputeModuleGenerator(module,
                                                     groups_info[inner_point_idx][1],
                                                     fake_module)
                
            stage[name] = pipe_module
            
        return stage
            