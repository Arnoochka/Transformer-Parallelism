from mytransformers.parallel.pipeline_parallel.layers import (FakeModule, InnerStrategyModule, FinalStrategyModule)
from .BoundaryPointModuleGenerator import InnerBoundaryPointModuleGenerator
from .ComputeModuleGenerator import ComputeModuleGenerator
from typing import List, Tuple, Callable
from torch.nn import Module, ModuleList, ModuleDict
from torch.distributed import ProcessGroup
from mytransformers.parallel.moe_parallel.moe_pipeline.pipeline import Pipeline
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.moe_parallel.moe_pipeline.pipeline import BaseScheduler
    
    
class PipelineGenerator(ParallelModuleGenerator):
    """
    Генератор конвейерного параллелизма
    
    Args:
        model (Module): модель с подмененными слоями
        modules (ModuleList): модули, которые получены из get_stage
        final_strategy (FinalStrategyModule): финальная стратегия для актуализации данных на GPU
        groups_info (Tuple[ProcessGroup, Tuple[int]]): информация о группах стадий
        final_comm_group (ProcessGroup): финальная группа для передачи данных
        fake_args (Callable): генератор аргументов для FakeModule
    """
    
    def __new__(cls,
                model: Module,
                modules: ModuleList,
                final_strategy: FinalStrategyModule,
                final_comm_group: ProcessGroup,
                fake_args: Callable,) -> Module:
        pipeline = Pipeline(model.forward,
                            modules,
                            final_strategy,
                            final_comm_group,
                            fake_args)
        
        model.forward = pipeline.forward
        return model
    
    @staticmethod
    def get_stage(modules: List[Tuple[str, Module, FakeModule]],
                  inner_boundary_points: List[int],
                  groups_info: List[Tuple[ProcessGroup, List[int]]],
                  inner_comm_groups: List[ProcessGroup],
                  inner_strategies: List[InnerStrategyModule],
                  scheduler: BaseScheduler) -> ModuleDict:
        """
        Генерирует фактическую стадию

        Args:
            modules (List[Tuple[str, Module, FakeModule]]): (название исходного модуля, исходный модуль, модуль на подмену)
            inner_boundary_points (List[int]): индексы внутренних точек (модуей), где начинается (заверщается) "фактическая" стадия
            inner_groups_info (List[Tuple[ProcessGroup, List[int]]]): информация о группах процессов стадий
            inner_comm_groups (List[ProcessGroup]): внутренние группы для передачи данных
            inner_strategies (List[InnerStrategyModule]): внутренние стратегии передачи данных
            scheduler (BaseScheduler): расписание для коллективных операций
        """
        
        stage = ModuleDict()
        inner_point_idx = 0
        for idx, (name, module, fake_module) in enumerate(modules):
            if idx in inner_boundary_points:
                pipe_module = InnerBoundaryPointModuleGenerator(module,
                                                                groups_info[inner_point_idx],
                                                                groups_info[inner_point_idx + 1],
                                                                inner_comm_groups[inner_point_idx],
                                                                fake_module,
                                                                inner_strategies[inner_point_idx],
                                                                scheduler)
                inner_point_idx += 1
            else:
                pipe_module = ComputeModuleGenerator(module,
                                                     groups_info[inner_point_idx][1],
                                                     fake_module)
                
            stage[name] = pipe_module
            
        return stage
            