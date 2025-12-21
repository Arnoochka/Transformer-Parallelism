from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import (
    FakeModule, FakeTensorModule, FakeTupleTensorModule,
    StrategyModule, LeaderTupleStrategyModule, LeaderStrategyDictModule)
from typing import List, Tuple, Callable, Any, Dict
from torch.distributed import ProcessGroup
from transformers import OPTForCausalLM
import torch
from torch.nn import Module, ModuleList, ModuleDict

class OPTGenerator(ParallelModuleGenerator):
    def __new__(cls,
                module: OPTForCausalLM,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                comm_groups: List[ProcessGroup],
                embed_size: int,
                vocab_size: int,
                device: torch.device) -> Module:
        
        decoder = module.model.decoder
        num_layers = len(decoder.layers)
        
        orig_modules = OPTGenerator.get_orig_modules(module)
        fake_modules = OPTGenerator.get_fake_modules(num_layers, device)
        modules = [(name, orig_module, fake_module)
                   for (name, orig_module), fake_module in zip(orig_modules, fake_modules)]
        num_modules_per_stage = len(modules) // num_stages
        inner_boundary_points = [num_modules_per_stage * k - 1 for k in range(num_stages)][1:]
        
        inner_strategies = OPTGenerator.get_inner_strategies(len(inner_boundary_points))
        
        stage: ModuleDict = PipelineGenerator.get_stage(modules,
                                                        inner_boundary_points,
                                                        groups_info,
                                                        comm_groups[:-1],
                                                        inner_strategies)
        
        module = OPTGenerator.replace_modules(module, stage)
        modules = [module for name, module in stage.items()]
        fake_args = OPTGenerator.build_fake_args(num_layers, embed_size, vocab_size)
        pipeline = PipelineGenerator(module,
                                     modules,
                                     LeaderStrategyDictModule(send_leader=1, recv_leader=0),
                                     groups_info,
                                     comm_groups[-1],
                                     fake_args)
        return pipeline.to(device)
        
    @staticmethod
    def get_fake_modules(num_layers: int, device: torch.device) -> List[FakeModule]:
        first_modules = [FakeTensorModule(device), FakeTensorModule(device)]
        inner_modules = [FakeTupleTensorModule(device) for _ in range(num_layers)]
        last_modules = [FakeTensorModule(device), FakeTensorModule(device)]
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
            setattr(module.model.decoder, name, stage[name])
        setattr(module, 'lm_head', stage['lm_head'])
        layers = ModuleList()
        for name, pipe_module in stage.items():
            if "layers" in name.split("-"):
                layers.append(pipe_module)
        module.model.decoder.layers = layers
        return module        
         
    @staticmethod
    def get_inner_strategies(num_points: int) -> List[StrategyModule]:
        return [LeaderTupleStrategyModule() for _ in range(num_points)]
    
    @staticmethod
    def build_fake_args(num_layers: int, embed_size: int, vocab_size: int) -> Callable:
        def _get_fake_args(mbatch_data: Dict) -> List[Any]:
            
            b, s = mbatch_data['input_ids'].size()
            first_args = [(b, s, embed_size), (b, s, embed_size)] 
            last_args = [(b, s, embed_size), (b, s, vocab_size)] 
            inner_args = [[(b, s, embed_size)] for _ in range(num_layers)]
            return [(fake_args,)  for fake_args in first_args + inner_args + last_args]
        
        return _get_fake_args