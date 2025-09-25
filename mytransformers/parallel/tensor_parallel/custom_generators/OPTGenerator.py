import torch
from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from transformers import OPTForCausalLM
from mytransformers.parallel.tensor_parallel.tp_generators import (TPModuleGenerator,
                                                                   TPColumnLinearGenerator,
                                                                   TPRowLinearGenerator,
                                                                   TPColumnEmbeddingGenerator,
                                                                   TPLayerNormGenerator)
from mytransformers.parallel.Reshaper import SimpleSplitter
from torch.nn import ModuleList, Linear, LayerNorm, Embedding

class OPTGenerator(TPModuleGenerator):
    def __new__(cls, module: OPTForCausalLM, tp_group: ProcessGroup, device: torch.device) -> OPTForCausalLM:
        TPColumnEmbeddingGenerator.use_all_gather = False
        TPLayerNormGenerator.use_all_gather = False
        TPColumnLinearGenerator.use_all_gather = True
        TPRowLinearGenerator.use_all_reduce = True
        module.lm_head = TPRowLinearGenerator(module.lm_head, tp_group, device)
        decoder = module.model.decoder
        for name, child in decoder.named_children():
            if isinstance(child, ModuleList):
                child = ModuleList([
                    OPTDecoderLayerGenerator(layer, tp_group, device)
                    for layer in child])
            elif type(child) is Embedding:
                child = TPColumnEmbeddingGenerator(child, tp_group, device)
            elif name == "embed_positions":
                child = SimpleSplitter(child, -1, tp_group).to(device)
            elif name == "final_layer_norm":
                child = TPLayerNormGenerator(child, tp_group, device)
            else:
                child = child.to(device)
            setattr(decoder, name, child)
        module.model.decoder = decoder
        return module
        
class OPTDecoderLayerGenerator(TPModuleGenerator):
    def __new__(cls, module: Module, tp_group: ProcessGroup, device: torch.device) -> Module:
        TPColumnLinearGenerator.use_all_gather = False
        TPRowLinearGenerator.use_all_reduce = True
        TPLayerNormGenerator.use_all_gather = False
        for name, child in module.named_children():
            if isinstance(child, Linear):
                if name == 'fc1':
                    child = TPRowLinearGenerator(child, tp_group, device)
                else:
                    child = TPColumnLinearGenerator(child, tp_group, device)
            elif isinstance(child, LayerNorm):
                child = TPLayerNormGenerator(child, tp_group, device)
            elif name == "activation_fn":
                child = child.to(device)
            else:
                child = OPTAttentionGenerator(child, tp_group, device)
            setattr(module, name, child)
        return module
        
class OPTAttentionGenerator(TPModuleGenerator):
    def __new__(cls, module: Module, tp_group: ProcessGroup, device: torch.device) -> Module:
        TPColumnLinearGenerator.use_all_gather = False
        TPRowLinearGenerator.use_all_reduce = True
        for name, child in module.named_children():
            if isinstance(child, Linear):
                if name == "out_proj":
                    child = TPColumnLinearGenerator(child, tp_group, device)
                else:
                    child = TPRowLinearGenerator(child, tp_group, device)
            setattr(module, name, child)
        world_size = dist.get_world_size(tp_group)
        assert module.num_heads % world_size == 0,\
            f"It is not possible to split query heads into {world_size} devices"
        return module

    

                
                
            