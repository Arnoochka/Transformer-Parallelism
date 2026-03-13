from mytransformers.parallel.pipeline_parallel_1.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel_1.layers.fake_modules import (
    FakeModule, FakeTensorModule, FakeTupleOptionalTensorModule)
from mytransformers.parallel.pipeline_parallel_1.layers.strategies import LeaderTupleStrategyModule
from typing import List, Tuple
from torch.distributed import ProcessGroup
from transformers import DebertaV2Model
import torch
from torch.nn import Module, ModuleList
from torch import LongTensor, FloatTensor
from transformers.cache_utils import Cache

class DebertaGenerator(ParallelModuleGenerator):
    batch_size: int
    seq_len: int
    def __new__(cls,
                module: DebertaV2Model,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                device: torch.device) -> Module:
        stages_info = [[] for _ in range(num_stages)]

        encoder = module.encoder
        layers = encoder.layer
        num_layers_per_stage = len(layers) // num_stages
        for k in range(num_stages - 1):
            start = num_layers_per_stage * k
            end = num_layers_per_stage * (k + 1) - 4
            stages_info[k] = [layer for layer in layers[start:end]]
            
        stages_info[-1] = [layer for layer in layers[end:]]
            
        stages_info[0] = [module.embeddings, encoder.conv] + stages_info[0]
        stages = [stages_info[idx] for idx in range(num_stages)]
        stages_fake_generators = DebertaGenerator.get_stages_fake_modules(stages, device)
        pipeline = PipelineGenerator(stages,
                                     groups_info,
                                     stages_fake_generators,
                                     LeaderTupleStrategyModule,
                                     device)
        setattr(module, 'embeddings', pipeline[0][0])
        setattr(encoder, 'conv', pipeline[0][1])
        pipeline[0] = pipeline[0][2:]
        layers = ModuleList(sum(pipeline, []))
        setattr(encoder, 'layer', layers)
        module.encoder = encoder
        
        return module.to(device)
    
    @staticmethod
    def get_stages_fake_modules(stages: List[List[Module]], device) -> List[List[FakeModule]]:
        batch_size = DebertaGenerator.batch_size
        seq_len = DebertaGenerator.seq_len
        embed_size = 1536
        return [[FakeTensorModule((batch_size, seq_len, embed_size), device),
                FakeTensorModule((batch_size, seq_len, embed_size), device)] + \
                    [FakeTupleOptionalTensorModule([(batch_size, seq_len, embed_size), None], device)
                     for _ in range(len(stages[0][2:]))]] + \
                         [[FakeTupleOptionalTensorModule([(batch_size, seq_len, embed_size), None], device)
                           for _ in range(len(stage))] for stage in stages[1:]]
                
                
                    
                