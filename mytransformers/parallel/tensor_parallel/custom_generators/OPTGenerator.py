import torch
from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from transformers import OPTForCausalLM
from mytransformers.parallel.tensor_parallel.tp_generators import (TPModuleGenerator,
                                                                   TPColumnLinearGenerator,
                                                                   TPRowLinearGenerator)
from torch.nn import ModuleList, Linear, LayerNorm

class OPTGenerator(TPModuleGenerator):
    def __new__(cls, module: OPTForCausalLM, tp_group: ProcessGroup) -> OPTForCausalLM:
        TPColumnLinearGenerator.use_all_gather = True
        device = torch.cuda.current_device()
        module.lm_head = TPColumnLinearGenerator(module.lm_head, tp_group)
        decoder = module.model.decoder
        for name, child in decoder.named_children():
            if isinstance(child, Linear):
                child = TPColumnLinearGenerator(child, tp_group)
            elif isinstance(child, ModuleList):
                child = ModuleList([
                    OPTDecoderLayerGenerator(layer, tp_group)
                    for layer in child])
            else:
                child = child.to(device)
            setattr(decoder, name, child)
        module.model.decoder = decoder
        return module
        
class OPTDecoderLayerGenerator(TPModuleGenerator):
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        TPColumnLinearGenerator.use_all_gather = False
        TPRowLinearGenerator.use_all_reduce = True
        device = torch.cuda.current_device()
        for name, child in module.named_children():
            if isinstance(child, Linear):
                if name == 'fc1':
                    child = TPColumnLinearGenerator(child, tp_group)
                else:
                    child = TPRowLinearGenerator(child, tp_group)
            elif isinstance(child, LayerNorm) or name == "activation_fn":
                child = child.to(device)
            else:
                child = OPTAttentionGenerator(child, tp_group)
            setattr(module, name, child)
        return module
        
class OPTAttentionGenerator(TPModuleGenerator):
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        TPColumnLinearGenerator.use_all_gather = False
        TPRowLinearGenerator.use_all_reduce = True
        device = torch.cuda.current_device()
        for name, child in module.named_children():
            if isinstance(child, Linear):
                if name == "out_proj":
                    child = TPRowLinearGenerator(child, tp_group)
                else:
                    child = TPColumnLinearGenerator(child, tp_group)
            setattr(module, name, child)
        world_size = dist.get_world_size(tp_group)
        assert module.num_heads % world_size == 0,\
            f"It is not possible to split query heads into {world_size} devices"
        module.num_heads = module.num_heads // world_size
        module.embed_dim = module.embed_dim // world_size
        return module
                
                
            