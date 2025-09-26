from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import PipeRole, PipeDummyModule
from .PipeStrategyGenerators import (
    PipeStrategyGenerator,
    PipeLeaderStrategyGenerator,
    PipeBroadcastLeaderStrategyGenerator
    )
from .PipeComputeModuleGenerator import PipeComputeModuleGenerator
from typing import List, Tuple, Optional
from torch.nn import Module, ModuleList
from torch.distributed import ProcessGroup
import torch


class PipelineGenerator(ParallelModuleGenerator):
    strategy_gen: PipeStrategyGenerator = PipeLeaderStrategyGenerator
    final_strategy_gen: PipeStrategyGenerator = PipeBroadcastLeaderStrategyGenerator
    compute_gen: PipeComputeModuleGenerator = PipeComputeModuleGenerator
    groups_info: List[Tuple[ProcessGroup, List[int]]] = None
    stages: List[List[Module]] = None
    final_recv_group: Optional[ProcessGroup] = None

    def __new__(cls, module: Module, device: torch.device) -> List[ModuleList]:
        if len(cls.groups_info) != len(cls.stages):
            raise AttributeError("the number of groups does not match the number of stages")

        pipeline = []
        
        cls.compute_gen.role = PipeRole.compute
        cls.compute_gen.neighbor_role = PipeRole.dummy
        cls.compute_gen.neighbor_module = PipeDummyModule(device)
        cls.strategy_gen.role = PipeRole.computeAndSend
        cls.strategy_gen.neighbor_role = PipeRole.recv
        cls.strategy_gen.neighbor_module = PipeDummyModule(device)
        cls.final_strategy_gen.role = PipeRole.computeAndSend
        cls.final_strategy_gen.neighbor_role = PipeRole.recv
        cls.final_strategy_gen.neighbor_module = PipeDummyModule(device)
        
        for idx, (group, ranks) in enumerate(cls.groups_info[:-1]):
            modules = cls.stages[idx]
            next_group, next_ranks = cls.groups_info[idx + 1]
            
            cls.compute_gen.group_ranks = ranks
            cls.compute_gen.next_group_ranks = next_ranks
            cls.strategy_gen.group_info = (group, ranks)
            cls.strategy_gen.next_group_info = (next_group, next_ranks)

            pipeline.append(cls.get_stage(modules=modules, device=device, last_stage=False))
            
        final_group, final_ranks = cls.groups_info[-1]
        cls.compute_gen.group_ranks = final_ranks
        cls.compute_gen.next_group_ranks = []
        cls.final_strategy_gen.group_info = (final_group, final_ranks)
        final_modules = cls.stages[-1]
        pipeline.append(cls.get_stage(modules=final_modules, device=device, last_stage=True))
        
        return pipeline

    @classmethod
    def get_stage(cls,
                  modules: List[Module],
                  device: torch.device,
                  last_stage: bool) -> ModuleList:
        stage = ModuleList()
        
        for module in modules[:-1]:
            stage.append(cls.compute_gen(module, device))

        last_module = modules[-1]
        if not last_stage:
            stage.append(cls.strategy_gen(last_module, device))
            
        else:
            stage.append(cls.final_strategy_gen(last_module, device))

        return stage
