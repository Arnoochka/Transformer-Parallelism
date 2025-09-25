from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from .PipeModules import PipeRole, PipeModule
from .strategies import StrategyGenerator, LeaderStrategyGenerator, Strategy, CommRole
from typing import Callable, List, Tuple, Optional
from torch.nn import Module, ModuleList, Sequential
from torch.distributed import ProcessGroup
import torch.distributed as dist
import torch

class PipeModuleGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: Module,
                role: PipeRole,
                in_stage: bool,
                neibhor_role: PipeRole,
                neibhor_module: Callable,
                in_next_stage: bool,
                device: torch.device) -> PipeModule:
        
        if in_stage: 
            return PipeModule(role, module).to(device)
        else:
            if in_next_stage:
                return PipeModule(neibhor_role, neibhor_module).to(device)
            else:
                return PipeModule(PipeRole.dummy, lambda x: x).to(device)

Stage = ModuleList
class PipeStageGenerator(ParallelModuleGenerator):
    last_stage: bool = False
    def __new__(cls,
                modules: ModuleList,
                in_stage: bool,
                in_next_stage: bool,
                strategy: Strategy,
                device: torch.device) -> Stage:
        stage = Stage()
        for module in modules[:-1]:
            stage.append(PipeModuleGenerator(module,
                                             PipeRole.compute,
                                             in_stage,
                                             PipeRole.dummy,
                                             lambda x: x,
                                             in_next_stage,
                                             device))
        if not cls.last_stage:
            stage.append(PipeModuleGenerator(Sequential(modules[-1], strategy),
                                             PipeRole.computeAndcomm,
                                             in_stage,
                                             PipeRole.comm,
                                             strategy,
                                             in_next_stage,
                                             device))
        else:
            stage.append(PipeModuleGenerator(modules[-1],
                                             PipeRole.compute,
                                             in_stage,
                                             PipeRole.dummy,
                                             lambda x: x,
                                             in_next_stage,
                                             device))
        return stage
    
class PipelineGenerator(ParallelModuleGenerator):
    strategy: StrategyGenerator = LeaderStrategyGenerator
    def __new__(cls,
                stages: ModuleList[Stage],
                groups: List[Tuple[ProcessGroup, List[int]]],
                device: torch.device) -> ModuleList[Stage]:
        pipeline = ModuleList()
        rank = dist.get_rank()
        for idx, (group, ranks) in enumerate(groups[:-1]):
            modules = stages[idx]
            in_stage = rank in ranks
            next_group, next_ranks = groups[idx + 1]
            in_next_stage = rank in next_ranks
            comm_role = CommRole.none
            if in_stage: comm_role = CommRole.send
            elif in_next_stage: comm_role = CommRole.recv
            strategy = cls.strategy(comm_role, group, next_group)
            pipeline.append(PipeStageGenerator(modules,
                                               in_stage,
                                               in_next_stage,
                                               strategy,
                                               device))
            
            
        
        
        
        
        
                    
                    
                    
        