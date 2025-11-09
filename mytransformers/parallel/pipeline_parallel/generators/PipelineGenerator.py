from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import (
    PipeRole, FakeModule, LeaderStrategyModule, LeaderTupleStrategyModule)
from .layer_generators import StrategyModuleGenerator, ComputeModuleGenerator
from .PipeModuleGenerator import PipeModuleGenerator
from typing import List, Tuple, Dict
from torch.nn import Module
from torch.distributed import ProcessGroup
import torch


class PipelineGenerator:
    
    """
    Генератор стадий конвейерного параллелизма
    
    Args:
        stages (List[List[Module]]): Слои, разделенные ао стадиям
        groups_info (List[Tuple[ProcessGroup, List[int]]]): информация о группах
        stages_fake_modules (List[List[FakeModule]]): фейковые модули для подмены модулей
        device (torch.device): текущее устройство
    """
    
    def __new__(cls,
                stages: List[List[Module]],
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                stages_fake_modules: List[List[FakeModule]],
                device: torch.device) -> List[List[Module]]:
        if len(groups_info) != len(stages):
            raise AttributeError("the number of groups does not match the number of stages")
        pipeline = []
        for idx, (group, ranks) in enumerate(groups_info):
            is_last_stage = (idx == len(stages) - 1)
            stage = stages[idx]
            stage_fake_modules = stages_fake_modules[idx]
            compute_kwargs = {
                "role": PipeRole.compute,
                "group_ranks": ranks}
            strategy_kwargs = {
                "role": PipeRole.computeAndSend,
                "group_info": (group, ranks),
                "next_role": PipeRole.recv,
                "next_module": stage_fake_modules[-1],
                "next_group_info": groups_info[0] if is_last_stage else groups_info[idx + 1],
                "strategy": LeaderTupleStrategyModule}
                
            if is_last_stage:
                strategy_kwargs["strategy"] = LeaderStrategyModule
            new_stage = PipelineGenerator.get_stage(
                stage,
                stage_fake_modules,
                compute_kwargs,
                strategy_kwargs,
                device)
            
            pipeline.append(new_stage)
                
        return pipeline

    @classmethod
    def get_stage(cls,
                  modules: List[Module],
                  fake_modules: List[FakeModule],
                  gen_kwargs: Dict,
                  last_gen_kwargs: Dict,
                  device: torch.device) -> List[Module]:
        stage = []
        PipeModuleGenerator.generator = ComputeModuleGenerator
        PipeModuleGenerator.gen_kwargs = gen_kwargs
        for module, fake_module in zip(modules[:-1], fake_modules[:-1]):
            gen_kwargs['fake_module'] = fake_module
            stage.append(PipeModuleGenerator(module, device))

        last_gen_kwargs['fake_module'] = fake_modules[-1]
        PipeModuleGenerator.generator = StrategyModuleGenerator
        PipeModuleGenerator.gen_kwargs = last_gen_kwargs
        stage.append(PipeModuleGenerator(modules[-1], device))

        return stage
