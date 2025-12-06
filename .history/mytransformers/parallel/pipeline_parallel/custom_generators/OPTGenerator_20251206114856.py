from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator, StageGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import (
    FakeModule, FakeTensorModule, FakeTupleTensorModule,
    StrategyModule, LeaderStrategyModule)
from typing import List, Tuple, Callable, Any
from torch.distributed import ProcessGroup
from transformers import OPTForCausalLM
import torch
from torch.nn import Module, ModuleList, ModuleDict
from torch import Tensor

class OPTGenerator(ParallelModuleGenerator):
    num_stages: int = 2
    groups_info: List[Tuple[ProcessGroup, List[int]]]
    bcast_groups: Tuple[ProcessGroup, ProcessGroup]
    num_microbatches: int
    def __new__(cls,
                module: OPTForCausalLM,
                device: torch.device) -> Module:
        decoder = module.model.decoder
        num_layers = len(decoder.layers)
        num_layers_per_stage = len(decoder.layers) // cls.num_stages
        inner_boundary_groups = [num_layers_per_stage * k for k in range(cls.num_stages)][1:]
        
        
    @staticmethod
    def get_fake_modules(num_layers: int, device: torch.device) -> List[FakeModule]:
        first_modules = [FakeTupleTensorModule[device], FakeTupleTensorModule[device]]
        inner_modules = [FakeTensorModule(device) for _ in range(num_layers)]
        last_modules = [FakeTupleTensorModule[device], FakeTupleTensorModule[device]]
        return first_modules + inner_modules + last_modules
    
    @staticmethod
    def get_orig_modules(module: OPTForCausalLM) -> List[Tuple[str, Module]]:
        decoder = module.model.decoder
        first_modules = [('embed_tokens', decoder.embed_tokens),
                         ('embed_positions', decoder.embed_positions)]
        inner_modules = [(f"layers-{name}", layer) for name, layer in enumerate(decoder.layers)]
        last_modules = [('final_layer_norm', decoder.final_layer_norm),
                        ('lm_head', module.lm_head)]
        return first_modules + inner_modules + last_modules
    
    @staticmethod
    def get_strategies(num_points: int) -> List[StrategyModule]:
        return [LeaderStrategyModule() for _ in range(num_points + 2)]
    
    @staticmethod
    def build_fake_args() -> Callable:
        def _get_fake_args(input_ids: Tensor, **kwargs) -> List[]:
                       