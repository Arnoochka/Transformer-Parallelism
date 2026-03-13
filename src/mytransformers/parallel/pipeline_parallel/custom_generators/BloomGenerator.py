from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import (
    FakeModule, FakeTensorModule, FakeTupleTensorWithCacheModule,
    StrategyModule, LeaderTupleStrategyModule, FinalStrategyDictModule)
from typing import List, Tuple, Callable, Any, Dict
from torch.distributed import ProcessGroup
from transformers import BloomForCausalLM
import torch
from torch.nn import Module, ModuleList, ModuleDict

class BloomGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: BloomForCausalLM,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                inner_comm_groups: List[ProcessGroup],
                final_comm_group: ProcessGroup,
                embed_size: int,
                vocab_size: int,
                device: torch.device) -> Module:
        
        transformer = module.transformer
        num_layers = len(transformer.h)
        
        orig_modules = BloomGenerator.get_orig_modules(module)
        fake_modules = BloomGenerator.get_fake_modules(num_layers, device)
        modules = [(name, orig_module, fake_module)
                   for (name, orig_module), fake_module in zip(orig_modules, fake_modules)]
        num_modules_per_stage = len(modules) // num_stages
        inner_boundary_points = [num_modules_per_stage * k - 1 for k in range(num_stages)][1:]
        
        inner_strategies = BloomGenerator.get_inner_strategies(len(inner_boundary_points))
        
        stage: ModuleDict = PipelineGenerator.get_stage(modules,
                                                        inner_boundary_points,
                                                        groups_info,
                                                        inner_comm_groups,
                                                        inner_strategies)
        
        module = BloomGenerator.replace_modules(module, stage)
        modules = [module for name, module in stage.items()]
        fake_args = BloomGenerator.build_fake_args(num_layers, embed_size, vocab_size)
        pipeline = PipelineGenerator(module,
                                     modules,
                                     FinalStrategyDictModule(send_rank=num_stages-1),
                                     final_comm_group,
                                     fake_args)
        return pipeline.to(device)
        
    @staticmethod
    def get_fake_modules(num_layers: int, device: torch.device) -> List[FakeModule]:
        first_modules = [FakeTensorModule(device), FakeTensorModule(device)]
        inner_modules = [FakeTupleTensorWithCacheModule(device, cache_name='layer_past') for _ in range(num_layers)]
        last_modules = [FakeTensorModule(device), FakeTensorModule(device)]
        return first_modules + inner_modules + last_modules
    
    @staticmethod
    def get_orig_modules(module: BloomForCausalLM) -> List[Tuple[str, Module]]:
        transformer = module.transformer
        first_modules = [('word_embeddings', transformer.word_embeddings),
                         ('word_embeddings_layernorm', transformer.word_embeddings_layernorm)]
        inner_modules = [(f"layers-{name}", layer) for name, layer in enumerate(transformer.h)]
        last_modules = [('ln_f', transformer.ln_f),
                        ('lm_head', module.lm_head)]
        return first_modules + inner_modules + last_modules
    
    @staticmethod
    def replace_modules(module: BloomForCausalLM, stage: ModuleDict) -> Module:
        for name in ['word_embeddings', 'word_embeddings_layernorm', 'ln_f']:
            setattr(module.transformer, name, stage[name])
        setattr(module, 'lm_head', stage['lm_head'])
        layers = ModuleList()
        for name, pipe_module in stage.items():
            if "layers" in name.split("-"):
                layers.append(pipe_module)
        module.transformer.h = layers
        return module        
         
    @staticmethod
    def get_inner_strategies(num_points: int) -> List[StrategyModule]:
        return [LeaderTupleStrategyModule() for _ in range(num_points)]
    
    @staticmethod
    def build_fake_args(num_layers: int, embed_size: int, vocab_size: int) -> Callable:
        def _get_fake_args(mbatch_data: Dict) -> List[Any]:
            b, s = mbatch_data['input_ids'].size()
            first_args = [((b, s, embed_size),), ((b, s, embed_size),)] 
            last_args = [((b, s, embed_size),), ((b, s, vocab_size),)] 
            inner_args = [([(b, s, embed_size)],) for _ in range(num_layers)]
            return first_args + inner_args + last_args
        
        return _get_fake_args
    
    