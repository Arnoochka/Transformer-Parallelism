import torch
from torch.nn import Module
from torch import Tensor
from typing import Optional
from .SimpleLayers import AddNorm
from .Attention import SelfAttention, CrossAttention
from .MoE import MoELayer
from .SimpleLayers import FeedForward

from enum import Enum

class TransformerType(Enum):
    Encoder = 0
    Decoder = 1
    EncoderDecoder = 2
    
class FFNType(Enum):
    MoE = MoELayer
    FFN = FeedForward
        
        
class TransformerDecoderLayer(Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.masked_attn = SelfAttention(config)
        
        self.encoder_output = config.transformer_type == TransformerType.EncoderDecoder
        if self.encoder_output:
            self.cross_attn = CrossAttention(config)
            self.cross_attn_norm = AddNorm(config)
            
        self.masked_attn_norm = AddNorm(config)
        self.logits_norm = AddNorm(config)
        
        self.ffn = config.ffn_type.value(config)
        
    def forward(self,
                x: Tensor,
                encoder_output: Optional[Tensor] = None,
                decoder_mask: Optional[Tensor] = None) -> Tensor:
        
        attn_out = self.masked_attn_norm(x, self.masked_attn(x, mask=decoder_mask))
        if self.encoder_output:
            attn_out = self.cross_attn_norm(
                attn_out, self.cross_attn(attn_out, encoder_output)
                )
            
        logits = self.logits_norm(attn_out, self.ffn(attn_out))
        return logits
    
class TransformerEncoderLayer(Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.attn = SelfAttention(config)
            
        self.attn_norm = AddNorm(config)
        self.logits_norm = AddNorm(config)
        
        self.ffn = config.ffn_type.value(config)
        
    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        attn_out = self.attn_norm(x, self.attn(x, mask))
        logits = self.logits_norm(attn_out, self.ffn(attn_out))
        return logits