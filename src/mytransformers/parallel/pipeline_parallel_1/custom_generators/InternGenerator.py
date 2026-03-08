from mytransformers.parallel.pipeline_parallel_1.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel_1.layers.fake_modules import (
    FakeModule, FakeTensorModule, FakeTupleOptionalTensorModule)
from mytransformers.parallel.pipeline_parallel_1.layers.strategies import FinalStrategyModule
from typing import List, Tuple
from torch.distributed import ProcessGroup
from transformers import AutoModel
import torch
from torch.nn import Module, ModuleList

class InternGenerator(ParallelModuleGenerator):
    batch_size: int
    def __new__(cls,
                module: AutoModel,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                comm_groups: List[ProcessGroup],
                final_group_info: Tuple[ProcessGroup, List[int]],
                final_comm_group: ProcessGroup,
                device: torch.device) -> Module:
        stages_info = [[] for _ in range(num_stages)]

        encoder = module.encoder
        layers = encoder.layers
        num_layers_per_stage = len(layers) // num_stages
        for k in range(num_stages - 1):
            start = num_layers_per_stage * k
            end = num_layers_per_stage * (k + 1)
            stages_info[k] = [layer for layer in layers[start:end]]
            
        stages_info[-1] = [layer for layer in layers[end:]]
            
        stages_info[0] = [module.embeddings] + stages_info[0]
        stages = [stages_info[idx] for idx in range(num_stages)]
        stages_fake_generators = InternGenerator.get_stages_fake_modules(stages, device)
        pipeline = PipelineGenerator(stages=stages,
                                     groups_info=groups_info,
                                     final_group_info=final_group_info,
                                     stages_fake_modules=stages_fake_generators,
                                     final_strategy=FinalStrategyModule,
                                     device=device,
                                     comm_groups=comm_groups,
                                     final_comm_group=final_comm_group)
        setattr(module, 'embeddings', pipeline[0][0])
        pipeline[0] = pipeline[0][1:]
        layers = ModuleList(sum(pipeline, []))
        setattr(encoder, 'layers', layers)
        module.encoder = encoder
        
        return module.to(device)
    
    @staticmethod
    def get_stages_fake_modules(stages: List[List[Module]], device) -> List[List[FakeModule]]:
        batch_size = InternGenerator.batch_size
        seq_len = 1025
        embed_size = 3200
        return [[FakeTensorModule((batch_size, seq_len, embed_size), device)] + \
            [FakeTensorModule((batch_size, seq_len, embed_size), device)
             for _ in range(len(stages[0][1:]))]] + \
                 [[FakeTensorModule((batch_size, seq_len, embed_size), device)
                   for _ in range(len(stage))] for stage in stages[1:]] 
                
                
                    
                