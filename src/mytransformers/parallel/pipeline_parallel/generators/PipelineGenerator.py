from mytransformers.parallel.pipeline_parallel.layers import (FakeModule, StrategyModule)
from .BoundaryPointModuleGenerator import BoundaryPointModuleGenerator
from .ComputeModuleGenerator import ComputeModuleGenerator
from typing import List, Tuple, Callable
from torch.nn import Module, ModuleList, ModuleDict
from torch.distributed import ProcessGroup
import torch.distributed as dist
from mytransformers.parallel.pipeline_parallel.pipeline import Pipeline
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
    
    
class PipelineGenerator(ParallelModuleGenerator):
    """
    Генератор конвейерного параллелизма
    
    Args:
        model (Module): модель с подмененными слоями
        modules (ModuleList): модули, которые получены из get_stage
        final_strategy (StrategyModule): финальная стратегия для актуализации данных на GPU
        groups_info (Tuple[ProcessGroup, Tuple[int]]): информация о группах стадий
        final_comm_group (ProcessGroup): финальная группа для передачи данных
        fake_args (Callable): генератор аргументов для FakeModule
        
    """
    def __new__(cls,
                model: Module,
                modules: ModuleList,
                final_strategy: StrategyModule,
                groups_info: Tuple[ProcessGroup, Tuple[int]],
                final_comm_group: ProcessGroup,
                fake_args: Callable) -> Module:
        rank = dist.get_rank()
        for (curr_group, curr_ranks) in groups_info:
            if rank in curr_ranks: break
            
        final_group, final_ranks = groups_info[-1]
        final_strategy_args = (
            True if rank in final_ranks else False,
            curr_group,
            final_comm_group)
        
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
                  inner_comm_groups: List[ProcessGroup],
                  inner_strategies: List[StrategyModule]) -> ModuleDict:
        """
        Генерирует фактическую стадию

        Args:
            modules (List[Tuple[str, Module, FakeModule]]): (название исходного модуля, исходный модуль, модуль на подмену)
            inner_boundary_points (List[int]): индексы внутренних точек (модуей), где начинается (заверщается) "фактическая" стадия
            inner_groups_info (List[Tuple[ProcessGroup, List[int]]]): информация о группах процессов стадий
            inner_comm_groups (List[ProcessGroup]): внутренние группы для передачи данных
            inner_strategies (List[StrategyModule]): внутренние стратегии передачи данных
        """
        
        stage = ModuleDict()
        inner_point_idx = 0
        for idx, (name, module, fake_module) in enumerate(modules):
            if idx in inner_boundary_points:
                pipe_module = BoundaryPointModuleGenerator(module,
                                                           groups_info[inner_point_idx],
                                                           groups_info[inner_point_idx + 1],
                                                           inner_comm_groups[inner_point_idx],
                                                           fake_module,
                                                           inner_strategies[inner_point_idx])
                inner_point_idx += 1
            else:
                pipe_module = ComputeModuleGenerator(module,
                                                     groups_info[inner_point_idx][1],
                                                     fake_module)
                
            stage[name] = pipe_module
            
        return stage
            