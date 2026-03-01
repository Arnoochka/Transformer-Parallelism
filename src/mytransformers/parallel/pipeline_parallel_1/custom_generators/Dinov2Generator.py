from mytransformers.parallel.pipeline_parallel_1.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel_1.layers.fake_modules import (
    FakeModule, FakeTensorModule, FakeTupleOptionalTensorModule)
from mytransformers.parallel.pipeline_parallel_1.layers.strategies import LeaderStrategyModule
from typing import List, Tuple
from torch.distributed import ProcessGroup
from transformers import Dinov2Model
import torch
from torch.nn import Module, ModuleList

class Dinov2Generator(ParallelModuleGenerator):
    batch_size: int
    def __new__(cls,
                module: Dinov2Model,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                device: torch.device) -> Module:
        stages_info = [[] for _ in range(num_stages)]

        encoder = module.encoder
        layers = encoder.layer
        num_layers_per_stage = len(layers) // num_stages
        for k in range(num_stages - 1):
            start = num_layers_per_stage * k
            end = num_layers_per_stage * (k + 1)
            stages_info[k] = [layer for layer in layers[start:end]]
            
        stages_info[-1] = [layer for layer in layers[end:]]
            
        stages_info[0] = [module.embeddings] + stages_info[0]
        stages_info[-1] = stages_info[-1] + [module.layernorm]
        stages = [stages_info[idx] for idx in range(num_stages)]
        stages_fake_generators = Dinov2Generator.get_stages_fake_modules(stages, device)
        pipeline = PipelineGenerator(stages,
                                     groups_info,
                                     stages_fake_generators,
                                     LeaderStrategyModule,
                                     device)
        setattr(module, 'embeddings', pipeline[0][0])
        setattr(encoder, 'layernorm', pipeline[-1][-1])
        pipeline[0] = pipeline[0][1:]
        pipeline[-1] = pipeline[-1][:-1]
        layers = ModuleList(sum(pipeline, []))
        setattr(encoder, 'layer', layers)
        module.encoder = encoder
        
        return module.to(device)
    
    @staticmethod
    def get_stages_fake_modules(stages: List[List[Module]], device) -> List[List[FakeModule]]:
        batch_size = Dinov2Generator.batch_size
        seq_len = 257
        embed_size = 1536
        return [[FakeTensorModule((batch_size, seq_len, embed_size), device)] + \
            [FakeTupleOptionalTensorModule([(batch_size, seq_len, embed_size)], device)
             for _ in range(len(stages[0][2:]))]] + \
                 [[FakeTupleOptionalTensorModule([(batch_size, seq_len, embed_size)], device)
                   for _ in range(len(stage))] for stage in stages[1:-1]] + \
                       [[FakeTupleOptionalTensorModule([(batch_size, seq_len, embed_size)], device)
                         for _ in range(len(stages[-1][:-1]))] + [FakeTensorModule((batch_size, seq_len, embed_size), device)]]
                
                
                    
                