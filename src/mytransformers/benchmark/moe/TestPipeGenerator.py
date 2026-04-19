from .TestModel import TestModel
import torch
from torch import nn
from typing import List, Tuple, Callable, Any, Dict
from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import (
    FakeModule, FakeTensorModule, InnerStrategyModule, LeaderStrategyModule, FinalStrategyDictModule)
from torch.distributed import ProcessGroup

class TestPipeGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: TestModel,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                inner_comm_groups: List[ProcessGroup],
                final_comm_group: ProcessGroup,
                embed_size: int,
                vocab_size: int,
                device: torch.device) -> nn.Module:
        
        num_layers = len(module.layers)
        
        orig_modules = TestPipeGenerator.get_orig_modules(module)
        fake_modules = TestPipeGenerator.get_fake_modules(num_layers, device)
        modules = [(name, orig_module, fake_module)
                   for (name, orig_module), fake_module in zip(orig_modules, fake_modules)]
        num_modules_per_stage = len(modules) // num_stages
        inner_boundary_points = [num_modules_per_stage * k - 1 for k in range(num_stages)][1:]
        
        inner_strategies = TestPipeGenerator.get_inner_strategies(len(inner_boundary_points))
        
        stage: nn.ModuleDict = PipelineGenerator.get_stage(modules,
                                                           inner_boundary_points,
                                                           groups_info,
                                                           inner_comm_groups,
                                                           inner_strategies)
        
        module = TestPipeGenerator.replace_modules(module, stage)
        modules = [module for name, module in stage.items()]
        fake_args = TestPipeGenerator.build_fake_args(num_layers, embed_size, vocab_size)
        pipeline = PipelineGenerator(module,
                                     modules,
                                     FinalStrategyDictModule(send_rank=num_stages-1),
                                     final_comm_group,
                                     fake_args)
        return pipeline.to(device)
        
    @staticmethod
    def get_fake_modules(num_layers: int, device: torch.device) -> List[FakeModule]:
        return [FakeTensorModule(device)] +\
            [FakeTensorModule(device) for _ in range(num_layers)] +\
                [FakeTensorModule(device)]
    
    @staticmethod
    def get_orig_modules(module: TestModel) -> List[Tuple[str, nn.Module]]:
        return [("embed_tokens", module.embed_tokens)] +\
            [(f"layers-{idx}", layer) for idx, layer in enumerate(module.layers)] +\
                [('lm_head', module.lm_head)]
    
    @staticmethod
    def replace_modules(module: TestModel, stage: nn.ModuleDict) -> nn.Module:
        setattr(module, 'embed_tokens', stage['embed_tokens'])
        setattr(module, 'lm_head', stage['lm_head'])
        layers = nn.ModuleList()
        for name, pipe_module in stage.items():
            if "layers" in name.split("-"):
                layers.append(pipe_module)
        module.layers = layers
        return module        
         
    @staticmethod
    def get_inner_strategies(num_points: int) -> List[InnerStrategyModule]:
        return [LeaderStrategyModule() for _ in range(num_points)]
    
    @staticmethod
    def build_fake_args(num_layers: int, embed_size: int, vocab_size: int) -> Callable:
        def _get_fake_args(mbatch_data: Dict) -> List[Any]:
            b, s = mbatch_data['input_ids'].size()
            return [((b, s, embed_size),)] +\
                [((b, s, embed_size),) for _ in range(num_layers)] +\
                    [((b, s, vocab_size),)] 
        
        return _get_fake_args