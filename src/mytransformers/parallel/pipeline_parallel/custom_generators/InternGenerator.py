from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import (
    FakeModule, FakeTensorModule,
    StrategyModule, LeaderStrategyModule, FinalStrategyDictModule)
from typing import List, Tuple, Callable, Any, Dict
from torch.distributed import ProcessGroup
from transformers import AutoModel
import torch
from torch.nn import Module, ModuleList, ModuleDict

class InternGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: AutoModel,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                inner_comm_groups: List[ProcessGroup],
                final_comm_group: ProcessGroup,
                embed_size: int,
                vocab_size: int,
                device: torch.device) -> Module:
        
        encoder = module.encoder
        num_layers = len(encoder.layers)
        
        orig_modules = InternGenerator.get_orig_modules(module)
        fake_modules = InternGenerator.get_fake_modules(num_layers, device)
        modules = [(name, orig_module, fake_module)
                   for (name, orig_module), fake_module in zip(orig_modules, fake_modules)]
        num_modules_per_stage = len(modules) // num_stages
        inner_boundary_points = [num_modules_per_stage * k - 1 for k in range(num_stages)][1:]
        
        inner_strategies = InternGenerator.get_inner_strategies(len(inner_boundary_points))
        
        stage: ModuleDict = PipelineGenerator.get_stage(modules,
                                                        inner_boundary_points,
                                                        groups_info,
                                                        inner_comm_groups,
                                                        inner_strategies)
        
        module = InternGenerator.replace_modules(module, stage)
        modules = [module for name, module in stage.items()]
        fake_args = InternGenerator.build_fake_args(num_layers, embed_size, vocab_size)
        pipeline = PipelineGenerator(module,
                                     modules,
                                     FinalStrategyDictModule(send_rank=num_stages-1),
                                     final_comm_group,
                                     fake_args)
        return pipeline.to(device)
        
    @staticmethod
    def get_fake_modules(num_layers: int, device: torch.device) -> List[FakeModule]:
        return [FakeTensorModule(device)] +\
            [FakeTensorModule(device) for _ in range(num_layers)]
    
    @staticmethod
    def get_orig_modules(module: AutoModel) -> List[Tuple[str, Module]]:
        encoder = module.encoder
        embeddings = module.embeddings
        return [('embeddings', embeddings)] + \
            [(f"layer-{i}", layer) for i, layer in enumerate(encoder.layers)]
    
    @staticmethod
    def replace_modules(module: AutoModel, stage: ModuleDict) -> Module:
        setattr(module, 'embeddings', stage['embeddings'])
        layers = ModuleList()
        for name, pipe_module in stage.items():
            if "layer" in name.split("-"):
                layers.append(pipe_module)
        module.encoder.layers = layers
        return module        
         
    @staticmethod
    def get_inner_strategies(num_points: int) -> List[StrategyModule]:
        return [LeaderStrategyModule() for _ in range(num_points)]
    
    @staticmethod
    def build_fake_args(num_layers: int, embed_size: int, vocab_size: int) -> Callable:
        def _get_fake_args(mbatch_data: Dict) -> List[Any]:
            b, c, h, w = mbatch_data['pixel_values'].size()
            return [((b, 1025, embed_size),)] +\
                [((b, 1025, embed_size),) for _ in range(num_layers)]
        
        return _get_fake_args
    
    