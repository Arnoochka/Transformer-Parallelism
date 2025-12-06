from mytransformers.parallel.pipeline_parallel.layers import (FakeModule, StrategyModule)
from .BoundaryPointModuleGenerator import BoundaryPointModuleGenerator, MainBoundaryPointModuleGenerator
from .ComputeModuleGenerator import ComputeModuleGenerator
from .PipeGenerator import PipeGenerator
from typing import List, Tuple, Callable
from torch.nn import Module, ModuleList
from torch.distributed import ProcessGroup

from mytransformers.parallel.pipeline_parallel.Pipeline import Pipeline
    

class StageGenerator(PipeGenerator):
    """
    Генератор для фактической стадии
    
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
    def __new__(cls,
                model: Module,
                modules: List[Tuple[str, Module, FakeModule]],
                inner_boundary_points: List[int],
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                inner_strategies: List[StrategyModule],
                bcast_groups: Tuple[ProcessGroup, ProcessGroup],
                bcast_strategies: Tuple[StrategyModule, StrategyModule],
                fake_args: Callable,
                num_microbatches: int) -> Pipeline:
        
        stage = ModuleList()
        inner_point_idx = 0
        for idx, (name, module, fake_module) in enumerate(modules):
            if idx == 0:
                pipe_module = MainBoundaryPointModuleGenerator(module,
                                                               groups_info[0],
                                                               bcast_groups[0],
                                                               False,
                                                               bcast_strategies[0])
            elif idx == len(modules) - 1:
                pipe_module = MainBoundaryPointModuleGenerator(module,
                                                               groups_info[-1],
                                                               bcast_groups[-1],
                                                               True,
                                                               bcast_strategies[-1])
            elif idx in inner_boundary_points:
                pipe_module = BoundaryPointModuleGenerator(module,
                                                           groups_info[inner_point_idx],
                                                           inner_boundary_points[inner_point_idx + 1],
                                                           fake_module,
                                                           inner_strategies[inner_point_idx])
                inner_point_idx += 1
            else:
                pipe_module = ComputeModuleGenerator(module,
                                                     groups_info[inner_point_idx],
                                                     fake_module)
                
            stage.append(pipe_module)
            setattr(model, name, pipe_module)
            
        return Pipeline(model.forward, stage, fake_args, num_microbatches)
    
    
class PipelineGenerator(PipeGenerator):
    """
    Генератор конвейерного параллелизма
    
    Args:
        model (Module): исходная модель
        modules (List[Tuple[str, Module, FakeModule]]): модули на подмену, (имя, оригинальный модуль, фейковый модуль)
        fake_args (Callable): генератор аргкментов для "фейковых" слоев
        num_microbatches (int): число микробатчей
    """
    def __new__(cls,
                model: Module,
                modules: List[Tuple[str, Module, FakeModule]],
                fake_args: Callable,
                num_microbatches: int) -> Pipeline:
        
        pipeline = Pipeline(model.forward, modules, fake_args, num_microbatches)
        
            