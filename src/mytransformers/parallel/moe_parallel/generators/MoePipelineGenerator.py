import torch
from torch import Tensor
from typing import List, Tuple, Callable, Dict
from torch.nn import Module, ModuleList, ModuleDict
from torch.distributed import ProcessGroup
from mytransformers.parallel.pipeline_parallel.layers import FakeModule, InnerStrategyModule, FinalStrategyModule
from mytransformers.parallel.pipeline_parallel.generators import ComputeModuleGenerator
from .MoeLayerGenerator import MoeLayerGenerator
from mytransformers.parallel.moe_parallel.pipeline import MoePipeline, BaseScheduler, MoePipeInnerBoundaryPointModule
from mytransformers.parallel.moe_parallel.layers import MoeDPExperts, MoeSparseBlockModule
from mytransformers.parallel.moe_parallel.pipeline.MoePipeInnerBoundaryPointModule import MoePipeInnerBoundaryPointModule
from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from .MoeSparseBlockGenerator import MoeSparseBlockDPModuleGenerator
import torch.distributed as dist
from dataclasses import dataclass

@dataclass(frozen=True)
class MoeLayerConfig:
    module: Module 
    gate: Callable
    main_rank: int
    next_main_rank: int
    expert_idxs: List[Tensor]
    group: ProcessGroup
    
    
class MoePipelineGenerator(PipelineGenerator):
    def __new__(cls,
                model: Module,
                modules: ModuleList,
                inner_boundary_points: List[int],
                final_strategy: FinalStrategyModule,
                final_comm_group: ProcessGroup,
                fake_args: Callable,
                scheduler: BaseScheduler) -> Module:
        
        pipeline = MoePipeline(model.forward,
                               modules,
                               inner_boundary_points,
                               final_strategy,
                               final_comm_group,
                               fake_args,
                               scheduler=scheduler)
        
        model.forward = pipeline.forward
        return model
    
    @staticmethod
    def get_stage(modules: List[Tuple[str, Module, FakeModule, bool]],
                  moe_configs: Dict[str, MoeLayerConfig],
                  k: int,
                  groups_info: List[Tuple[ProcessGroup, List[int]]],
                  inner_boundary_points: List[int],
                  scheduler: BaseScheduler,
                  device: torch.device) -> ModuleDict:
        
        stage = ModuleDict()
        inner_point_idx = 0
        for idx, (name, module, fake_module, is_moe) in enumerate(modules):
            if is_moe:
                moe_config: MoeLayerConfig = moe_configs[name]
                moe = MoeSparseBlockDPModuleGenerator(module=moe_config.module,
                                                      gate=moe_config.gate,
                                                      k=k,
                                                      main_rank=moe_config.main_rank,
                                                      next_main_rank=moe_config.next_main_rank,
                                                      replace_experts_layer=MoeDPExperts,
                                                      expert_idxs=moe_config.expert_idxs,
                                                      moe_group=moe_config.group,
                                                      scheduler=scheduler,
                                                      device=device)
                
                moe_module = MoeLayerGenerator(module,
                                               moe,
                                               groups_info[inner_point_idx][1],
                                               fake_module,
                                               device)
            else:
                moe_module = ComputeModuleGenerator(module,
                                                    groups_info[inner_point_idx][1],
                                                    fake_module)

            stage[name] = moe_module
            
            if idx in inner_boundary_points:
                inner_point_idx += 1
                
        return stage