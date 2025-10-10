from mytransformers.parallel.pipeline_parallel.generators import PipelineGenerator
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
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
        stages_info[k] = stages_info[0] + layers[start:end]
            
        stages_info[0] = [decoder.embed_tokens, decoder.embed_positions] + stages_info[0]
        stages_info[-1] = stages_info[-1] + [decoder.final_layer_norm]
        stages = [stages_info[idx] for idx in range(cls.num_stages)]
        pipeline = PipelineGenerator(stages, cls.groups_info, device)
        setattr(decoder, 'embed_tokens', pipeline[0][0])
        setattr(decoder, 'embed_positions', pipeline[0][1])
        setattr(decoder, 'final_layer_norm', pipeline[-1][-1])
        pipeline[0] = pipeline[0][2:]
        pipeline[-1] = pipeline[-1][:-1]
        layers = ModuleList(sum(pipeline, []))
        setattr(decoder, 'layers', layers)
        module.model.decoder = decoder
        
        return module
        
                
                
                    
                