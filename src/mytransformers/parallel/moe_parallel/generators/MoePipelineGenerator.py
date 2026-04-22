from mytransformers.parallel.pipeline_parallel.layers import (FakeModule, InnerStrategyModule, FinalStrategyModule)
from typing import List, Tuple, Callable, Dict
from torch.nn import Module, ModuleList, ModuleDict
from torch.distributed import ProcessGroup
import torch.distributed as dist
import torch
from torch import Tensor
from mytransformers.parallel.moe_parallel.pipeline.MoePipeline import MoePipeline
from mytransformers.parallel.pipeline_parallel.pipeline.Pipeline import Pipeline
from mytransformers.parallel.pipeline_parallel.layers import PipeInnerBoundaryPointModule
from mytransformers.parallel.moe_parallel.pipeline.MoePipeInnerBoundaryPointModule import MoePipeInnerBoundaryPointModule
from mytransformers.parallel.moe_parallel.layers import MoeExperts, MoeDPExperts
from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from .MoeSperseBlockGenerator import MoeSparseBlockDPModuleGenerator
from mytransformers.parallel.moe_parallel.pipeline.Scheduler import BaseScheduler
    
    
class MoePipelineGenerator(PipelineGenerator):
    def __new__(cls,
                model: Module,
                modules: ModuleList,
                final_strategy: FinalStrategyModule,
                final_comm_group: ProcessGroup,
                fake_args: Callable,
                scheduler: BaseScheduler) -> Module:
        pipeline = MoePipeline(model.forward,
                               modules,
                               final_strategy,
                               final_comm_group,
                               fake_args,
                               scheduler=scheduler)
        
        model.forward = pipeline.forward
        return model
        
        return model
    
    @staticmethod
    def get_stage(modules: List[Tuple[str, Module, FakeModule]],
                  inner_boundary_points: List[int],
                  groups_info: List[Tuple[ProcessGroup, List[int]]],
                  inner_comm_groups: List[ProcessGroup],
                  inner_strategies: List[InnerStrategyModule],
                  is_moe: List[bool],
                  moe_layer_configs: Dict[Tuple[Callable, int, List[Tensor], ProcessGroup]],
                  scheduler: BaseScheduler,
                  dim_buffer: Tensor,
                  device: torch.device) -> ModuleDict:
        stage = super().get_stage(modules,
                                  inner_boundary_points,
                                  groups_info,
                                  inner_comm_groups,
                                  inner_strategies)
        moe_stage = ModuleDict()
        for idx, (name, module) in enumerate(stage.items()):
            moe_module = module
            module.modu
            if is_moe[idx]:
                gate, main_rank, expert_idxs, moe_group = moe_layer_configs[f"layer-{name}"]
                fake_module = moe_module.module if isinstance(moe_module.module, FakeModule) else None
                moe_module.module = MoeSparseBlockDPModuleGenerator(module=moe_module.module,
                                                                    gate=gate,
                                                                    main_rank=main_rank,
                                                                    replace_experts_layer=MoeDPExperts,
                                                                    expert_idxs=expert_idxs,
                                                                    moe_group=moe_group,
                                                                    dim_buffer=torch.empty_like(dim_buffer),
                                                                    scheduler=scheduler,
                                                                    device=device,
                                                                    fake_module=fake_module)
            if isinstance(moe_module, PipeInnerBoundaryPointModule):
                moe_module = MoePipeInnerBoundaryPointModule(role=moe_module.role,
                                                             module=moe_module,
                                                             current_group=moe_module.current_group,
                                                             comm_group=moe_module.comm_group,
                                                             strategy=moe_module.strategy,
                                                             scheduler=scheduler)
            moe_stage[name] = moe_module
                
                    
                    
                    
                
                
            