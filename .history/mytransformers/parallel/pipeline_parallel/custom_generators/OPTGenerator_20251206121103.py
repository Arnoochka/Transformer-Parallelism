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
    num_stages: int
    groups_info: List[Tuple[ProcessGroup, List[int]]]
    bcast_groups: Tuple[ProcessGroup, ProcessGroup]
    num_microbatches: int
    def __new__(cls,
                module: OPTForCausalLM,
                device: torch.device) -> Module:
        decoder = module.model.decoder
        num_layers = len(decoder.layers)
        num_layers_per_stage = len(decoder.layers) // cls.num_stages
        inner_boundary_points = [num_layers_per_stage * k for k in range(cls.num_stages)][1:]
        
        orig_modules = OPTGenerator.get_orig_modules(module)
        fake_modules = OPTGenerator.get_fake_modules(num_layers, device)
        modules = [(name, orig_module, fake_module)
                   for (name, orig_module, fake_module) in zip(orig_modules, fake_modules)]
        
        strategies = OPTGenerator.get_strategies(len(inner_boundary_points))
        bcast_stretegies = (strategies[0], strategies[-1])
        inner_strategies = strategies[1:-1]
        
        stage: ModuleDict = StageGenerator(modules,
                                           inner_boundary_points,
                                           cls.groups_info,
                                           inner_strategies,
                                           cls.bcast_groups,
                                           bcast_stretegies)
        module = OPTGenerator.replace_modules(module, stage)
        modules = [module for name, module in stage.items()]
        fake_args = OPTGenerator.build_fake_args(num_layers)
        pipeline = PipelineGenerator(module,
                                     modules, 
                                     fake_args,
                                     cls.num_microbatches)
        return pipeline.to(device)
        
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
    def replace_modules(module: OPTForCausalLM, stage: ModuleDict) -> Module:
        for name in ['embed_tokens', 'embed_positions', 'final_layer_norm']:
            setattr(module, name, stage[name])
        setattr(module, 'lm_head', stage['lm_head'])
        layers = ModuleList()
        for name, pipe_module in stage.items():
            if "layers" in name.split("-"):
                layers.append(pipe_module)
        module.model.decoder.layers = layers
        return module        
         
    @staticmethod
    def get_strategies(num_points: int) -> List[StrategyModule]:
        return [LeaderStrategyModule() for _ in range(num_points + 2)]
    
    @staticmethod
    def build_fake_args(num_layers: int) -> Callable:
        def _get_fake_args(input_ids: Tensor, **kwargs) -> List[Any]:
            b, s, h = input_ids.size()
            first_args = [[(b, s, h)], [(b, s, h)]] 
            last_args = [[(b, s, h)], [(b, s, h)]] 
            inner_args = [(b, s, h) for _ in range(num_layers)]
            return first_args + inner_args + last_args
        
        return _get_fake_args