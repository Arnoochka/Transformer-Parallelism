from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.fake_output_generators import (
    FakeGenerator, FakeSeqGenerator, FakeTupleTensorGenerator)
from typing import List, Tuple
from torch.distributed import ProcessGroup
from transformers import OPTForCausalLM
import torch
from torch.nn import Module, ModuleList

class OPTGenerator(ParallelModuleGenerator):
    num_stages: int = 2
    groups_info: List[Tuple[ProcessGroup, List[int]]]
    def __new__(cls,
                module: OPTForCausalLM,
                device: torch.device) -> Module:
        stages_info = [[] for _ in range(cls.num_stages)]

        decoder = module.model.decoder
        layers = decoder.layers
        num_layers_per_stage = len(layers) // cls.num_stages
        for k in range(cls.num_stages):
            start = num_layers_per_stage * k
            end = num_layers_per_stage * (k + 1)
            stages_info[k] = [layer for layer in layers[start:end]]
            
        stages_info[0] = [decoder.embed_tokens, decoder.embed_positions] + stages_info[0]
        stages_info[-1] = stages_info[-1] + [decoder.final_layer_norm, module.lm_head]
        stages = [stages_info[idx] for idx in range(cls.num_stages)]
        stages_fake_generators = OPTGenerator.get_stages_fake_generators(stages, device)
        pipeline = PipelineGenerator(stages, cls.groups_info, stages_fake_generators, device)
        setattr(decoder, 'embed_tokens', pipeline[0][0])
        setattr(decoder, 'embed_positions', pipeline[0][1])
        setattr(decoder, 'final_layer_norm', pipeline[-1][-2])
        setattr(module, 'lm_head', pipeline[-1][-1])
        pipeline[0] = pipeline[0][2:]
        pipeline[-1] = pipeline[-1][:-2]
        layers = ModuleList(sum(pipeline, []))
        setattr(decoder, 'layers', layers)
        module.model.decoder = decoder
        
        return module.to(device)
    @staticmethod
    def get_stages_fake_generators(stages: List[List[Module]], device) -> List[List[FakeGenerator]]:
        # TODO: Delete this and fix problem with fake generators
        first_stage = [FakeSeqGenerator((12, 64, 2048), seq_dim=1, device=device), 
                       FakeSeqGenerator((12, 64, 2048), seq_dim=1, device=device)] + \
                          [FakeTupleTensorGenerator([(12, 64, 2048)], device=device)
                           for _ in range(len(stages[0][2:]))] 
        last_stage = [FakeTupleTensorGenerator([(12, 64, 2048)], device=device)
                           for _ in range(len(stages[-1][:-1]))] + \
                               [FakeSeqGenerator((12, 64, 2048), seq_dim=1, device=device)]
        fake_generators = [first_stage, last_stage]
        for stage in stages[1:-1]:
            fake_generators.append([FakeSeqGenerator((12, 64, 2048), seq_dim=1, device=device)
                                    for _ in range(len(stage))])
            
        return fake_generators
                
                
                    
                