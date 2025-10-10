from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import PipeRole, PipeDummyModule
from .PipeStrategyGenerators import LeaderStrategyGenerator
from .PipeModuleGenerator import PipeModuleGenerator
from .ComputeModuleGenerator import ComputeModuleGenerator
from typing import List, Tuple, Dict
from torch.nn import Module
from torch.distributed import ProcessGroup
import torch


class PipelineGenerator:
    strategy_gen = LeaderStrategyGenerator
    def __new__(cls,
                stages: List[List[Module]],
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                device: torch.device) -> List[List[Module]]:
        if len(groups_info) != len(stages):
            raise AttributeError("the number of groups does not match the number of stages")
        pipeline = []
        compute_kwargs = {"role": PipeRole.compute, "group_ranks": ranks}
        for idx, (group, ranks) in enumerate(groups_info[:-1]):
            modules = stages[idx]
            strategy_kwargs = {"role": PipeRole.computeAndSend,
                               "group_info": (group, ranks),
                               "next_role": PipeRole.recv,
                               "next_module": PipeDummyModule(device),
                               "next_group_info": groups_info[idx + 1],
                               "leader_rank": ranks[-1]}
            stage = PipelineGenerator.get_stage(
                modules,
                ComputeModuleGenerator,
                compute_kwargs,
                cls.strategy_gen if idx < len(stages) - 1 else ComputeModuleGenerator,
                strategy_kwargs if idx < len(stages) - 1 else compute_kwargs,
                device)
            
            pipeline.append(stage)
                
        return pipeline

    @classmethod
    def get_stage(cls,
                  modules: List[Module],
                  generator: ParallelModuleGenerator,
                  gen_kwargs: Dict,
                  last_generator: ParallelModuleGenerator,
                  last_gen_kwargs: Dict,
                  device: torch.device) -> List[Module]:
        stage = []
        
        for module in modules[:-1]:
            PipeModuleGenerator.generator = generator
            PipeModuleGenerator.gen_kwargs = gen_kwargs
            stage.append(PipeModuleGenerator(module, device))

        last_module = modules[-1]
        PipeModuleGenerator.generator = last_generator
        PipeModuleGenerator.gen_kwargs = last_gen_kwargs
        stage.append(PipeModuleGenerator(last_module, device))

        return stage
