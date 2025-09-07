from torch.distributed import ProcessGroup
from torch.nn import Module
import torch
from .TPModuleGenerator import TPModuleGenerator
from mytransformers import TransformerType
from .TPAttentionGenerator import TPCrossAttentionGenerator, TPSelfAttentionGenerator
from .TPFeedForwardGenerator import TPFeedForwardGenerator
from mytransformers.parallel.tensor_parallel.tp_layers import (
    TPTransformerEncoderLayer,
    TPTransformerDecoderLayer)
    
class TPTransformerDecoderLayerGenerator(TPModuleGenerator):
    config = None
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPTransformerDecoderLayer:
        TPSelfAttentionGenerator.config = cls.config
        TPCrossAttentionGenerator.config = cls.config
        
        use_encoder = cls.config.transformer_type == TransformerType.EncoderDecoder
        masked_attn = TPSelfAttentionGenerator(module.masked_attn, tp_group)
        ffn = TPFeedForwardGenerator(module.ffn, tp_group)
        device = torch.cuda.current_device()
        masked_attn_norm = module.masked_attn_norm.to(device)
        logits_norm = module.logits_norm.to(device)
        if use_encoder:
            cross_attn = TPCrossAttentionGenerator(module.cross_attn, tp_group)
            cross_attn_norm = module.cross_attn_norm.to(device) 
            return TPTransformerDecoderLayer(cls.config,
                                                   masked_attn,
                                                   masked_attn_norm,
                                                   cross_attn,
                                                   cross_attn_norm,
                                                   ffn,
                                                   logits_norm,
                                                   tp_group)
        else:
            return TPTransformerDecoderLayer(cls.config,
                                                   masked_attn,
                                                   masked_attn_norm,
                                                   None,
                                                   None,
                                                   ffn,
                                                   logits_norm,
                                                   tp_group)
        
        
class TPTransformerEncoderLayerGenerator(TPModuleGenerator):
    config = None
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPTransformerDecoderLayer:
        TPSelfAttentionGenerator.config = cls.config
        attn = TPSelfAttentionGenerator(module.attn, tp_group)
        ffn = TPFeedForwardGenerator(module.ffn, tp_group)
        device = torch.cuda.current_device()
        attn_norm = module.attn_norm.to(device)
        logits_norm = module.logits_norm.to(device)
        return TPTransformerEncoderLayer(cls.config,
                                               attn,
                                               attn_norm,
                                               ffn,
                                               logits_norm,
                                               tp_group)
        
        