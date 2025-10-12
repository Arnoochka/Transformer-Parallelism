from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import (
    PipeRole, PipeDummyModule, PipeLeaderStrategyModule, PipeLeaderTupleStrategyModule)
from mytransformers.parallel.pipeline_parallel.fake_output_generators import FakeGenerator
from .layer_generators import StrategyModuleGenerator, ComputeModuleGenerator
from .PipeModuleGenerator import PipeModuleGenerator
from typing import List, Tuple, Dict
from torch.nn import Module
from torch.distributed import ProcessGroup
import torch


class PipelineGenerator:
    def __new__(cls,
                stages: List[List[Module]],
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                stages_fake_generators: List[List[FakeGenerator]],
                device: torch.device) -> List[List[Module]]:
        if len(groups_info) != len(stages):
            raise AttributeError("the number of groups does not match the number of stages")
        pipeline = []
        StrategyModuleGenerator.strategy_module = PipeLeaderTupleStrategyModule
        for idx, (group, ranks) in enumerate(groups_info):
            is_last_stage = (idx == len(stages) - 1)
            stage = stages[idx]
            stage_fake_generators = stages_fake_generators[idx]
            compute_kwargs = {
                "role": PipeRole.compute,
                "group_ranks": ranks}
            strategy_kwargs = {
                "role": PipeRole.computeAndSend,
                "group_info": (group, ranks),
                "next_role": PipeRole.recv,
                "next_module": PipeDummyModule(stage_fake_generators[-1]),
                "next_group_info": groups_info[0] if is_last_stage else groups_info[idx + 1]}
                
            if is_last_stage:
                # TODO: Add async strategy for real pipeline parallel or add PipelineEngine
                StrategyModuleGenerator.strategy_module = PipeLeaderStrategyModule
                # strategy_kwargs = compute_kwargs
                
                
            new_stage = PipelineGenerator.get_stage(
                stage,
                stage_fake_generators,
                ComputeModuleGenerator,
                compute_kwargs,
                StrategyModuleGenerator, # if not is_last_stage else ComputeModuleGenerator,
                strategy_kwargs,
                device)
            
            pipeline.append(new_stage)
                
        return pipeline

    @classmethod
    def get_stage(cls,
                  modules: List[Module],
                  fake_generators: List[FakeGenerator],
                  generator: ParallelModuleGenerator,
                  gen_kwargs: Dict,
                  last_generator: ParallelModuleGenerator,
                  last_gen_kwargs: Dict,
                  device: torch.device) -> List[Module]:
        stage = []
        for module, fake_generator in zip(modules[:-1], fake_generators[:-1]):
            gen_kwargs['fake_generator'] = fake_generator
            PipeModuleGenerator.generator = generator
            PipeModuleGenerator.gen_kwargs = gen_kwargs
            stage.append(PipeModuleGenerator(module, device))

        last_gen_kwargs['fake_generator'] = fake_generators[-1]
        PipeModuleGenerator.generator = last_generator
        PipeModuleGenerator.gen_kwargs = last_gen_kwargs
        stage.append(PipeModuleGenerator(modules[-1], device))

        return stage
