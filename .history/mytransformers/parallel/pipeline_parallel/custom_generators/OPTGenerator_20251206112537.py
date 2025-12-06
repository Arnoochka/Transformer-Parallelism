from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers.fake_modules import ( FakeModule)
from typing import List, Tuple
from torch.distributed import ProcessGroup
from transformers import OPTForCausalLM
import torch
from torch.nn import Module, ModuleList

class OPTGenerator(ParallelModuleGenerator):
    num_stages: int = 2
    groups_info: List[Tuple[ProcessGroup, List[int]]]
    bcast_groups: Tuple[ProcessGroup, ProcessGroup]
    def __new__(cls,
                module: OPTForCausalLM,
                device: torch.device) -> Module:
        stages_info = [[] for _ in range(cls.num_stages)]

        decoder = module.model.decoder
        layers = decoder.layers
        num_layers_per_stage = len(layers) // cls.num_stages
                       