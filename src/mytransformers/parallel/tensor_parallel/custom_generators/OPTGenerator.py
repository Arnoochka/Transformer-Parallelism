import torch
from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from transformers import OPTForCausalLM
from mytransformers.parallel.tensor_parallel.generators import (TPColumnLinearGenerator,
                                                                   TPRowLinearGenerator,
                                                                   TPColumnEmbeddingGenerator,
                                                                   TPSplittedLayerNormGenerator)
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.Reshaper import SimpleSplitter
from torch.nn import ModuleList, Linear, Embedding

def set_tp_group(tp_group: ProcessGroup) -> None:
    """устанавливает группу для всех слоев"""
    TPColumnLinearGenerator.tp_group = tp_group
    TPRowLinearGenerator.tp_group = tp_group
    TPColumnEmbeddingGenerator.tp_group = tp_group
    TPSplittedLayerNormGenerator.tp_group = tp_group
    OPTGenerator.tp_group = tp_group
    OPTDecoderLayerGenerator.tp_group = tp_group
    OPTAttentionGenerator.tp_group = tp_group
    

class OPTGenerator(ParallelModuleGenerator):
    """
    кастомный генератор для OPTModel из huggingfacce
    """
    tp_group: ProcessGroup = None
    @torch.no_grad()
    def __new__(cls, module: OPTForCausalLM, device: torch.device) -> OPTForCausalLM:
        TPColumnEmbeddingGenerator.use_all_gather = True
        TPSplittedLayerNormGenerator.use_all_gather = False
        TPColumnLinearGenerator.use_all_gather = False
        TPRowLinearGenerator.use_all_reduce = True
        module.lm_head = TPColumnLinearGenerator(module.lm_head, device)
        decoder = module.model.decoder
        for name, child in decoder.named_children():
            if isinstance(child, ModuleList):
                child = ModuleList([
                    OPTDecoderLayerGenerator(layer, device)
                    for layer in child])
            elif type(child) is Embedding:
                child = TPColumnEmbeddingGenerator(child, device)
            elif name == "embed_positions":
                child = child.to(device)
            elif name == "final_layer_norm":
                child = child.to(device)
            else:
                child = child.to(device)
            setattr(decoder, name, child)
            
        TPColumnLinearGenerator.use_all_gather = True
        module.lm_head = TPColumnLinearGenerator(module.lm_head, device)
        module.model.decoder = decoder
        return module
        
class OPTDecoderLayerGenerator(ParallelModuleGenerator):
    tp_group: ProcessGroup = None
    def __new__(cls, module: Module, device: torch.device) -> Module:
        TPColumnLinearGenerator.use_all_gather = False
        TPRowLinearGenerator.use_all_reduce = True
        TPSplittedLayerNormGenerator.use_all_gather = True
        for name, child in module.named_children():
            if isinstance(child, Linear):
                if name == 'fc1':
                    child = TPColumnLinearGenerator(child, device)
                else:
                    child = TPRowLinearGenerator(child, device)
            elif name == "self_attn_layer_norm":
                child = child.to(device)
            elif name == "activation_fn" or name == "final_layer_norm":
                child = child.to(device)
            else:
                child = OPTAttentionGenerator(child, device)
            setattr(module, name, child)
        return module
        
class OPTAttentionGenerator(ParallelModuleGenerator):
    tp_group: ProcessGroup = None
    def __new__(cls, module: Module, device: torch.device) -> Module:
        TPColumnLinearGenerator.use_all_gather = False
        TPRowLinearGenerator.use_all_reduce = True
        for name, child in module.named_children():
            if isinstance(child, Linear):
                if name == "out_proj":
                    child = TPRowLinearGenerator(child, device)
                else:
                    child = TPColumnLinearGenerator(child, device)
            setattr(module, name, child)
        world_size = dist.get_world_size(cls.tp_group)
        assert module.num_heads % world_size == 0,\
            f"It is not possible to split query heads into {world_size} devices"
        module.num_heads = module.num_heads // world_size
        module.embed_dim = module.embed_dim // world_size
        
        return module

    

                
                
            