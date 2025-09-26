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

def set_tp_group(tp_group: ProcessGroup) -> None:
    TPModuleGenerator.tp_group = tp_group
    TPColumnLinearGenerator.tp_group = tp_group
    TPRowLinearGenerator.tp_group = tp_group
    TPColumnEmbeddingGenerator.tp_group = tp_group
    TPLayerNormGenerator.tp_group = tp_group
    OPTGenerator.tp_group = tp_group
    OPTDecoderLayerGenerator.tp_group = tp_group
    OPTAttentionGenerator.tp_group = tp_group
    

class OPTGenerator(TPModuleGenerator):
    tp_group: ProcessGroup = None
    @torch.no_grad()
    def __new__(cls, module: OPTForCausalLM, device: torch.device) -> OPTForCausalLM:
        TPColumnEmbeddingGenerator.use_all_gather = False
        TPLayerNormGenerator.use_all_gather = False
        TPColumnLinearGenerator.use_all_gather = True
        TPRowLinearGenerator.use_all_reduce = True
        module.lm_head = TPRowLinearGenerator(module.lm_head, device)
        decoder = module.model.decoder
        for name, child in decoder.named_children():
            if isinstance(child, ModuleList):
                child = ModuleList([
                    OPTDecoderLayerGenerator(layer, device)
                    for layer in child])
            elif type(child) is Embedding:
                child = TPColumnEmbeddingGenerator(child, device)
            elif name == "embed_positions":
                child = SimpleSplitter(child, -1, cls.tp_group).to(device)
            elif name == "final_layer_norm":
                child = TPLayerNormGenerator(child, device)
            else:
                child = child.to(device)
            setattr(decoder, name, child)
        module.model.decoder = decoder
        return module
        
class OPTDecoderLayerGenerator(TPModuleGenerator):
    tp_group: ProcessGroup = None
    def __new__(cls, module: Module, device: torch.device) -> Module:
        TPColumnLinearGenerator.use_all_gather = False
        TPRowLinearGenerator.use_all_reduce = True
        TPLayerNormGenerator.use_all_gather = False
        for name, child in module.named_children():
            if isinstance(child, Linear):
                if name == 'fc1':
                    child = TPRowLinearGenerator(child, device)
                else:
                    child = TPColumnLinearGenerator(child, device)
            elif isinstance(child, LayerNorm):
                child = TPLayerNormGenerator(child, device)
            elif name == "activation_fn":
                child = child.to(device)
            else:
                child = OPTAttentionGenerator(child, device)
            setattr(module, name, child)
        return module
        
class OPTAttentionGenerator(TPModuleGenerator):
    tp_group: ProcessGroup = None
    def __new__(cls, module: Module, device: torch.device) -> Module:
        TPColumnLinearGenerator.use_all_gather = False
        TPRowLinearGenerator.use_all_reduce = True
        for name, child in module.named_children():
            if isinstance(child, Linear):
                if name == "out_proj":
                    child = TPColumnLinearGenerator(child, device)
                else:
                    child = TPRowLinearGenerator(child, device)
            setattr(module, name, child)
        world_size = dist.get_world_size(cls.tp_group)
        assert module.num_heads % world_size == 0,\
            f"It is not possible to split query heads into {world_size} devices"
        return module

    

                
                
            