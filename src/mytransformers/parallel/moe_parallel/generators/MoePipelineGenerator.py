import torch
from torch import Tensor
from typing import List, Tuple, Callable, Dict
from torch.nn import Module, ModuleList, ModuleDict
from torch.distributed import ProcessGroup
from mytransformers.parallel.pipeline_parallel.layers import FakeModule, InnerStrategyModule, FinalStrategyModule
from mytransformers.parallel.pipeline_parallel.layers import PipeInnerBoundaryPointModule
from mytransformers.parallel.moe_parallel.pipeline import MoePipeline, BaseScheduler, MoePipeInnerBoundaryPointModule
from mytransformers.parallel.moe_parallel.layers import MoeDPExperts, MoeSparseBlockModule
from mytransformers.parallel.moe_parallel.pipeline.MoePipeInnerBoundaryPointModule import MoePipeInnerBoundaryPointModule
from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from .MoeSperseBlockGenerator import MoeSparseBlockDPModuleGenerator
import torch.distributed as dist
from dataclasses import dataclass

@dataclass(frozen=True)
class MoeLayerConfig:
    gate: Callable
    main_rank: int
    expert_idxs: List[Tensor]
    
    
class MoePipelineGenerator(PipelineGenerator):
    reset_modules: List[MoePipeInnerBoundaryPointModule | MoeSparseBlockModule] = None
    def __new__(cls,
                model: Module,
                modules: ModuleList,
                final_strategy: FinalStrategyModule,
                final_comm_group: ProcessGroup,
                fake_args: Callable,
                scheduler: BaseScheduler) -> Module:
        pipeline = MoePipeline(model.forward,
                               modules,
                               cls.reset_modules,
                               final_strategy,
                               final_comm_group,
                               fake_args,
                               scheduler=scheduler)
        
        model.forward = pipeline.forward
        return model
    
    @staticmethod
    def get_stage(modules: List[Tuple[str, Module, FakeModule]],
                  inner_boundary_points: List[int],
                  groups_info: List[Tuple[ProcessGroup, List[int]]],
                  inner_comm_groups: List[ProcessGroup],
                  inner_strategies: List[InnerStrategyModule],
                  is_moe: List[bool],
                  moe_layer_configs: Dict[str, MoeLayerConfig],
                  scheduler: BaseScheduler,
                  dim_buffer: Tensor,
                  moe_group: ProcessGroup,
                  device: torch.device) -> ModuleDict:
        pipe_stage = PipelineGenerator.get_stage(modules,
                                                 inner_boundary_points,
                                                 groups_info,
                                                 inner_comm_groups,
                                                 inner_strategies)
        moe_stage = ModuleDict()
        reset_modules = []
        for idx, (name, module, fake_module) in enumerate(modules):
            moe_module = pipe_stage[name]
            if is_moe[idx]:
                sparse_block = MoeSparseBlockDPModuleGenerator(module=module.moe,
                                                               gate=moe_layer_configs[name].gate,
                                                               main_rank=moe_layer_configs[name].main_rank,
                                                               replace_experts_layer=MoeDPExperts,
                                                               expert_idxs=moe_layer_configs[name].expert_idxs,
                                                               moe_group=moe_group,
                                                               dim_buffer=torch.empty_like(dim_buffer),
                                                               scheduler=scheduler,
                                                               device=device,
                                                               fake_module=fake_module)
                if dist.get_rank() ==moe_layer_configs[name].main_rank:
                    moe_module.module.moe = sparse_block
                else:
                    moe_module.module = sparse_block
                reset_modules.append(sparse_block)
            if isinstance(pipe_stage[name], PipeInnerBoundaryPointModule):
                moe_module = MoePipeInnerBoundaryPointModule(role=pipe_stage[name].role,
                                                             module=moe_module.module,
                                                             current_group=moe_module.current_group,
                                                             comm_group=moe_module.comm_group,
                                                             strategy=moe_module.strategy,
                                                             scheduler=scheduler)
                reset_modules.append(moe_module)
            moe_stage[name] = moe_module       
        MoePipelineGenerator.reset_modules = reset_modules
        return moe_stage