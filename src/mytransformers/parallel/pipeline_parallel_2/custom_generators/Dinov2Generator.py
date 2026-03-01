from mytransformers.parallel.pipeline_parallel_2.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel_2.layers import (
    FakeModule, FakeTensorModule, FakeTupleTensorModule,
    StrategyModule, LeaderTupleStrategyModule, LeaderStrategyDictModule)
from typing import List, Tuple, Callable, Any, Dict
from torch.distributed import ProcessGroup
from transformers import Dinov2Model
import torch
from torch.nn import Module, ModuleList, ModuleDict

class Dinov2Generator(ParallelModuleGenerator):
    def __new__(cls,
                module: Dinov2Model,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                comm_groups: List[ProcessGroup],
                embed_size: int,
                vocab_size: int,
                device: torch.device) -> Module:
        
        encoder = module.encoder
        num_layers = len(encoder.layer)
        
        orig_modules = Dinov2Generator.get_orig_modules(module)
        fake_modules = Dinov2Generator.get_fake_modules(num_layers, device)
        modules = [(name, orig_module, fake_module)
                   for (name, orig_module), fake_module in zip(orig_modules, fake_modules)]
        num_modules_per_stage = len(modules) // num_stages
        inner_boundary_points = [num_modules_per_stage * k - 1 for k in range(num_stages)][1:]
        
        inner_strategies = Dinov2Generator.get_inner_strategies(len(inner_boundary_points))
        
        stage: ModuleDict = PipelineGenerator.get_stage(modules,
                                                        inner_boundary_points,
                                                        groups_info,
                                                        comm_groups[:-1],
                                                        inner_strategies)
        
        module = Dinov2Generator.replace_modules(module, stage)
        modules = [module for name, module in stage.items()]
        fake_args = Dinov2Generator.build_fake_args(num_layers, embed_size, vocab_size)
        pipeline = PipelineGenerator(module,
                                     modules,
                                     LeaderStrategyDictModule(send_leader=num_stages - 1, recv_leader=0),
                                     groups_info,
                                     comm_groups[-1],
                                     fake_args)
        return pipeline.to(device)
        
    @staticmethod
    def get_fake_modules(num_layers: int, device: torch.device) -> List[FakeModule]:
        return [FakeTensorModule(device)] +\
            [FakeTupleTensorModule(device) for _ in range(num_layers)] +\
                [FakeTensorModule(device)]
    
    @staticmethod
    def get_orig_modules(module: Dinov2Model) -> List[Tuple[str, Module]]:
        encoder = module.encoder
        embeddings = module.embeddings
        layernorm = module.layernorm
        return [('embeddings', embeddings)] + \
            [(f"layer-{i}", layer) for i, layer in enumerate(encoder.layer)] + \
                [('layernorm', layernorm)]
    
    @staticmethod
    def replace_modules(module: Dinov2Model, stage: ModuleDict) -> Module:
        setattr(module, 'embeddings', stage['embeddings'])
        setattr(module, 'layernorm', stage['layernorm'])
        layers = ModuleList()
        for name, pipe_module in stage.items():
            if "layer" in name.split("-"):
                layers.append(pipe_module)
        module.encoder.layer = layers
        return module        
         
    @staticmethod
    def get_inner_strategies(num_points: int) -> List[StrategyModule]:
        return [LeaderTupleStrategyModule() for _ in range(num_points)]
    
    @staticmethod
    def build_fake_args(num_layers: int, embed_size: int, vocab_size: int) -> Callable:
        def _get_fake_args(mbatch_data: Dict) -> List[Any]:
            b, c, h, w = mbatch_data['pixel_values'].size()
            return [((b, 257, embed_size),)] +\
                [([(b, 257, embed_size)],) for _ in range(num_layers)] +\
                    [((b, 257, embed_size),)] 
        
        return _get_fake_args
    
    