from torch.distributed import ProcessGroup
from .ParallelModule import TensorParallelModule
import torch
from torch.nn import Module
import torch.nn as nn
from torch import Tensor
from typing import Optional

from enum import Enum

class TransformerType(Enum):
    Encoder = 0
    Decoder = 1
    EncoderDecoder = 2
    
class FFNType(Enum):
    MoE = 0
    FFN = 1
    
class AddNorm(Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_state,
                                 config.eps,
                                 config.elementwise_affine,
                                 config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))
        
        
class ParallelTransformerDecoderLayer(TensorParallelModule):
    def __init__(self,
                 config,
                 self_attn: TensorParallelModule,
                 cross_attn: Optional[TensorParallelModule],
                 ffn: TensorParallelModule,
                 tp_group: ProcessGroup) -> None:
        super().__init__(tp_group)
        
        self.masked_attn = self_attn
        
        self.encoder_output = config.transformer_type == TransformerType.EncoderDecoder
        if self.encoder_output:
            self.cross_attn = cross_attn
            self.cross_attn_norm = AddNorm(config)
            
        self.masked_attn_norm = AddNorm(config)
        self.logits_norm = AddNorm(config)
        
        self.ffn = ffn
        
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
                 attn: TensorParallelModule,
                 ffn: TensorParallelModule,
                 tp_group: ProcessGroup) -> None:
        super().__init__(tp_group)
        
        self.attn = attn
            
        self.attn_norm = AddNorm(config)
        self.logits_norm = AddNorm(config)
        
        self.ffn = ffn
        
    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        attn_out = self.attn_norm(x, self.attn(x, mask))
        logits = self.logits_norm(attn_out, self.ffn(attn_out))
        return logits
        
        