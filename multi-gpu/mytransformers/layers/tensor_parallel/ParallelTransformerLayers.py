from torch.distributed import ProcessGroup
import torch.distributed as dist
from torch.nn import Module
from torch.nn import Module
from .ParallelModule import TensorParallelModule, TensorParallelModuleGenerator
from ..SimpleLayers import AddNorm
from ..MoE import MoELayer
from ..TransformerLayers import TransformerType
from .ParallelAttention import ParallelCrossAttentionGenerator, ParallelSelfAttentionGenerator
from .ParallelFeedForward import ParallelFeedForwardGenerator
import torch
from torch import Tensor
from typing import Optional

from enum import Enum
    
class FFNType(Enum):
    MoE = MoELayer
    FFN = ParallelFeedForwardGenerator
        
        
class ParallelTransformerDecoderLayer(TensorParallelModule):
    def __init__(self,
                 config,
                 self_attn: Module,
                 cross_attn: Module,
                 ffn: Module,
                 tp_group: ProcessGroup) -> None:
        super().__init__(tp_group)
        
        self.masked_attn = ParallelSelfAttentionGenerator(self_attn, tp_group, config)
        
        self.encoder_output = config.transformer_type == TransformerType.EncoderDecoder
        if self.encoder_output:
            self.cross_attn = ParallelCrossAttentionGenerator(cross_attn, tp_group, config)
            self.cross_attn_norm = AddNorm(config)
            
        self.masked_attn_norm = AddNorm(config)
        self.logits_norm = AddNorm(config)
        
        self.ffn = config.ffn_type.value(ffn, tp_group, config)
        
    def forward(self,
                x: Tensor,
                encoder_output: Optional[Tensor] = None,
                decoder_mask: Optional[Tensor] = None) -> Tensor:
        
        seq_len = x.size(dim=1)
        if decoder_mask is None:
            decoder_mask = torch.tril(torch.ones(seq_len, seq_len)) \
                .unsqueeze(0).unsqueeze(1).to(x.device)
        
        attn_out = self.cross_attn_norm(x, self.masked_attn(x, mask=decoder_mask))
        if self.encoder_output:
            attn_out = self.masked_attn_norm(
                attn_out, self.cross_attn(attn_out, encoder_output)
                )
            
        logits = self.logits_norm(attn_out, self.ffn(attn_out))
        return logits
    
class ParallelTransformerEncoderLayer(TensorParallelModule):
    def __init__(self,
                 config,
                 attn: Tensor,
                 ffn: Module,
                 tp_group: ProcessGroup) -> None:
        super().__init__(tp_group)
        
        self.attn = ParallelSelfAttentionGenerator(attn, tp_group, config)
            
        self.attn_norm = AddNorm(config)
        self.logits_norm = AddNorm(config)
        
        self.ffn = config.ffn_type.value(ffn, tp_group, config)
        
    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        attn_out = self.attn_norm(x, self.attn(x, mask))
        logits = self.logits_norm(attn_out, self.ffn(attn_out))
        return logits
    
class ParallelTransformerDecoderGenerator(TensorParallelModuleGenerator):
    def __new__(cls, module: Module, tp_group: ProcessGroup, config) -> ParallelTransformerDecoderLayer:
        return ParallelTransformerDecoderLayer(config,
                                               module.masked_attn,
                                               module.cross_attn,
                                               module.ffn,
                                               tp_group)
        
class ParallelTransformerEncoderGenerator(TensorParallelModuleGenerator):
    def __new__(cls, module: Module, tp_group: ProcessGroup, config) -> ParallelTransformerEncoderLayer:
        return ParallelTransformerDecoderLayer(config,
                                               module.attn,
                                               module.ffn,
                                               tp_group)
        
        