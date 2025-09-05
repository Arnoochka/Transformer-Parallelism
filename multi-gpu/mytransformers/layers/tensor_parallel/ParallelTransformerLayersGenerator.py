from torch.distributed import ProcessGroup
from torch.nn import Module
import torch
from .ParallelModuleGenerator import TensorParallelModuleGenerator
from ..TransformerLayers import TransformerType
from .ParallelAttentionGenerator import ParallelCrossAttentionGenerator, ParallelSelfAttentionGenerator
from .ParallelFeedForwardGenerator import ParallelFeedForwardGenerator
from .parallel_layers import ParallelTransformerEncoderLayer, ParallelTransformerDecoderLayer
    
class ParallelTransformerDecoderGenerator(TensorParallelModuleGenerator):
    config = None
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelTransformerDecoderLayer:
        ParallelSelfAttentionGenerator.config = cls.config
        ParallelCrossAttentionGenerator.config = cls.config
        
        use_encoder = cls.config.transformer_type == TransformerType.EncoderDecoder
        masked_attn = ParallelSelfAttentionGenerator(module.masked_attn, tp_group)
        ffn = ParallelFeedForwardGenerator(module.ffn, tp_group)
        device = torch.cuda.current_device()
        masked_attn_norm = module.masked_attn_norm.to(device)
        logits_norm = module.logits_norm.to(device)
        if use_encoder:
            cross_attn = ParallelCrossAttentionGenerator(module.cross_attn, tp_group)
            cross_attn_norm = module.cross_attn_norm.to(device) 
            return ParallelTransformerDecoderLayer(cls.config,
                                                   masked_attn,
                                                   masked_attn_norm,
                                                   cross_attn,
                                                   cross_attn_norm,
                                                   ffn,
                                                   logits_norm,
                                                   tp_group)
        else:
            return ParallelTransformerDecoderLayer(cls.config,
                                                   masked_attn,
                                                   masked_attn_norm,
                                                   None,
                                                   None,
                                                   ffn,
                                                   logits_norm,
                                                   tp_group)
        
        
class ParallelTransformerEncoderGenerator(TensorParallelModuleGenerator):
    config = None
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelTransformerDecoderLayer:
        ParallelSelfAttentionGenerator.config = cls.config
        attn = ParallelSelfAttentionGenerator(module.attn, tp_group)
        ffn = ParallelFeedForwardGenerator(module.ffn, tp_group)
        device = torch.cuda.current_device()
        attn_norm = module.attn_norm.to(device)
        logits_norm = module.logits_norm.to(device)
        return ParallelTransformerEncoderLayer(cls.config,
                                               attn,
                                               attn_norm,
                                               ffn,
                                               logits_norm,
                                               tp_group)
        
        