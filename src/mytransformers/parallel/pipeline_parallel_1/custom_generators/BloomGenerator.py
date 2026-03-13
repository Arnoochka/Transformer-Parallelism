from mytransformers.parallel.pipeline_parallel_1.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel_1.layers.fake_modules import (
    FakeModule, FakeSeqModule, FakeTupleSeqModule)
from mytransformers.parallel.pipeline_parallel_1.layers.strategies import FinalStrategyModule
from typing import List, Tuple
from torch.distributed import ProcessGroup
from transformers import BloomForCausalLM
import torch
from torch.nn import Module, ModuleList
from torch import LongTensor, FloatTensor
from transformers.cache_utils import Cache

class BloomGenerator(ParallelModuleGenerator):
    batch_size: int
    seq_len: int
    def __new__(cls,
                module: BloomForCausalLM,
                num_stages: int,
                groups_info: List[Tuple[ProcessGroup, List[int]]],
                comm_groups: List[ProcessGroup],
                final_group_info: Tuple[ProcessGroup, List[int]],
                final_comm_group: ProcessGroup,
                device: torch.device) -> Module:
        stages_info = [[] for _ in range(num_stages)]

        transformer = module.transformer
        layers = transformer.h
        num_layers_per_stage = len(layers) // num_stages
        for k in range(num_stages):
            start = num_layers_per_stage * k
            end = num_layers_per_stage * (k + 1)
            stages_info[k] = [layer for layer in layers[start:end]]
            
        stages_info[0] = [transformer.word_embeddings, transformer.word_embeddings_layernorm] + stages_info[0]
        stages_info[-1] = stages_info[-1] + [transformer.ln_f, module.lm_head]
        stages = [stages_info[idx] for idx in range(num_stages)]
        stages_fake_generators = BloomGenerator.get_stages_fake_modules(stages, device)
        pipeline = PipelineGenerator(stages=stages,
                                     groups_info=groups_info,
                                     final_group_info=final_group_info,
                                     stages_fake_modules=stages_fake_generators,
                                     final_strategy=FinalStrategyModule,
                                     device=device,
                                     comm_groups=comm_groups,
                                     final_comm_group=final_comm_group)
        setattr(transformer, 'word_embeddings', pipeline[0][0])
        setattr(transformer, 'word_embeddings_layernorm', pipeline[0][1])
        setattr(transformer, 'ln_f', pipeline[-1][-2])
        setattr(module, 'lm_head', pipeline[-1][-1])
        pipeline[0] = pipeline[0][2:]
        pipeline[-1] = pipeline[-1][:-2]
        layers = ModuleList(sum(pipeline, []))
        setattr(transformer, 'h', layers)
        module.transformer = transformer
        
        return module.to(device)
    
    @staticmethod
    def get_stages_fake_modules(stages: List[List[Module]], device) -> List[List[FakeModule]]:
        batch_size = BloomGenerator.batch_size
        seq_len = BloomGenerator.seq_len
        embed_size = 4096
        vocab_size = 250880
        first_stage = [FakeSeqModule((batch_size, seq_len, embed_size), seq_dim=1, device=device), 
                       FakeSeqModule((batch_size, seq_len, embed_size), seq_dim=1, device=device)] + \
                          [FakeTupleSeqModule([(batch_size, seq_len, embed_size)], [1], device=device, cache_name='layer_past')
                           for _ in range(len(stages[0][2:]))] 
        last_stage = [FakeTupleSeqModule([(batch_size, seq_len, embed_size)], [1], device=device, cache_name='layer_past')
                           for _ in range(len(stages[-1][:-2]))] + \
                               [FakeSeqModule((batch_size, seq_len, embed_size), seq_dim=1, device=device),
                                FakeSeqModule((batch_size, seq_len, vocab_size), seq_dim=1, device=device)]
        fake_modules = [first_stage]
        for stage in stages[1:-1]:
            fake_modules.append([FakeTupleSeqModule([(batch_size, seq_len, embed_size)], [1], device=device, cache_name='layer_past')
                                    for _ in range(len(stage))])
        fake_modules.append(last_stage)  
        return fake_modules
                
                
                    
                