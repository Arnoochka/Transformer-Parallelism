from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import (
    FakeModule, FakeTensorModule, FakeTupleTensorModule,
    StrategyModule, LeaderStrategyModule)
from typing import List, Tuple
from torch.distributed import ProcessGroup
from transformers import OPTForCausalLM
import torch
from torch.nn import Module, ModuleList, ModuleDict

class OPTGenerator(ParallelModuleGenerator):
    num_stages: int = 2
    groups_info: List[Tuple[ProcessGroup, List[int]]]
    bcast_groups: Tuple[ProcessGroup, ProcessGroup]
    num_microbatches: int
    def __new__(cls,
                module: OPTForCausalLM,
                device: torch.device) -> Module:
        decoder = module.model.decoder
        layers = decoder.layers
        num_layers_per_stage = len(layers) // cls.num_stages
        inner_boundary_groups = [num_layers_per_stage * k for k in range(cls.num_stages)][1:]
        
                       