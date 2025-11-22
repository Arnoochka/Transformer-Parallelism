from mytransformers.parallel.pipeline_parallel.layers import (
    PipeRole, FakeModule, LeaderStrategyModule, LeaderTupleStrategyModule)
from mytransformers.parallel.ParallelModule import ParallelModule
from .BoundaryPointModuleGenerator import BoundaryPointModuleGenerator
from .ComputeModuleGenerator import ComputeModuleGenerator
from .PipeGenerator import PipeGenerator
from .PipeModuleGenerator import PipeModuleGenerator
from typing import List, Tuple, Dict, Callable
from torch.nn import Module
from torch.distributed import ProcessGroup

class StagesGenerator(PipeGenerator):
    
    """
    Генератор стадий конвейерного параллелизма
    
    Args:
        stages (List[List[Module]]): Слои, разделенные ао стадиям
        groups_info (List[Tuple[ProcessGroup, List[int]]]): информация о группах
        stages_fake_modules (List[List[FakeModule]]): фейковые модули для подмены модулей
    """
    
    def __new__(cls,
                stages_modules: List[List[Module]],
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                stages_fake_modules: List[List[FakeModule]]) -> List[List[Module]]:
        if len(groups_info) != len(stages_modules):
            raise AttributeError("the number of groups does not match the number of stages")
        stages = []
        for idx, (group, ranks) in enumerate(groups_info):
            is_last_stage = (idx == len(stages_modules) - 1)
            stage = stages_modules[idx]
            stage_fake_modules = stages_fake_modules[idx]
            compute_kwargs = {
                "group_ranks": ranks}
            strategy_kwargs = {
                "group_info": (group, ranks),
                "next_role": PipeRole.recv,
                "next_module": stage_fake_modules[-1],
                "next_group_info": groups_info[0] if is_last_stage else groups_info[idx + 1],
                "strategy": LeaderTupleStrategyModule}
                
            if is_last_stage:
                strategy_kwargs["strategy"] = LeaderStrategyModule
            new_stage = StagesGenerator.get_stage(
                stage,
                stage_fake_modules,
                compute_kwargs,
                strategy_kwargs)
            
            stages.append(new_stage)
                
        return stages

    @classmethod
    def get_stage(cls,
                  modules: List[Module],
                  fake_modules: List[FakeModule],
                  gen_kwargs: Dict,
                  last_gen_kwargs: Dict) -> List[Module]:
        stage = []
        PipeModuleGenerator.generator = ComputeModuleGenerator
        PipeModuleGenerator.gen_kwargs = gen_kwargs
        for module, fake_module in zip(modules[:-1], fake_modules[:-1]):
            gen_kwargs['fake_module'] = fake_module
            stage.append(ComputeModuleGenerator(module, **gen_kwargs))

        last_gen_kwargs['fake_module'] = fake_modules[-1]
        stage.append(BoundaryPointModuleGenerator(modules[-1], **last_gen_kwargs))

        return stage
    

class PipelineGenerator(PipeGenerator):
    def __new__(cls,
                module: Module,
                stages_info: List[List[str, Module, FakeModule]],
                groups_info: List[Tuple[ProcessGroup, List[int]]]) -> ParallelModule:
        get_info = lambda t: [[info[t] for info in stage_info]
                              for stage_info in stages_info]
        stages_names = get_info(0)
        stages_modules = get_info(1)
        stages_fake_modules = get_info(2)
        
        pipe_stages = StagesGenerator(stages_modules,
                                      groups_info,
                                      stages_fake_modules)
        
        
        
        for stage_names, pipe_stage in zip(stages_names, pipe_stages):
            for name, pipe_module in zip(stage_names, pipe_stage):
                setattr(module, name, pipe_module)
                
        return module
                
        
