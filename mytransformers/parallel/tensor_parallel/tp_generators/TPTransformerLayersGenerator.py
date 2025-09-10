from torch.distributed import ProcessGroup
from torch.nn import Module
import torch
from .TPModuleGenerator import TPModuleGenerator
from .TPAttentionGenerator import TPCrossAttentionGenerator, TPSelfAttentionGenerator
from .TPFeedForwardGenerator import TPFeedForwardGenerator
    
class TPTransformerDecoderLayerGenerator(TPModuleGenerator):
    self_attn_gen = TPSelfAttentionGenerator
    cross_attn_gen = TPCrossAttentionGenerator
    ffn_gen = TPFeedForwardGenerator
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        device = torch.cuda.current_device()
        use_encoder = module.encoder_output
        module.masked_attn = cls.self_attn_gen(module.masked_attn, tp_group)
        module.ffn = cls.ffn_gen(module.ffn, tp_group)
        module.masked_attn_norm = module.masked_attn_norm.to(device)
        module.logits_norm = module.logits_norm.to(device)
        if use_encoder:
            module.cross_attn = cls.cross_attn_gen(module.cross_attn, tp_group)
            module.cross_attn_norm = module.cross_attn_norm.to(device) 
        return module
        
        
class TPTransformerEncoderLayerGenerator(TPModuleGenerator):
    self_attn_gen = TPSelfAttentionGenerator
    ffn_gen = TPFeedForwardGenerator
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        module.attn = cls.self_attn_gen(module.attn, tp_group)
        module.ffn = cls.ffn_gen(module.ffn, tp_group)
        device = torch.cuda.current_device()
        module.attn_norm = module.attn_norm.to(device)
        module.logits_norm = module.logits_norm.to(device)
        return module
        
        