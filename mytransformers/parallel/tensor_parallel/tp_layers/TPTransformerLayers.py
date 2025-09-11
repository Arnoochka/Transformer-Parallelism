from torch.distributed import ProcessGroup
from .TPModule import TPModule
import torch
from torch.nn import Module
from torch import Tensor
from typing import Optional
from mytransformers.layers import TransformerType

        
class TPTransformerDecoderLayer(TPModule):
    def __init__(self,
                 config,
                 self_attn: TPModule,
                 self_attn_norm: Module,
                 cross_attn: Optional[TPModule],
                 cross_attn_norm: Optional[Module],
                 ffn: TPModule,
                 logits_norm: Module,
                 tp_group: ProcessGroup) -> None:
        super().__init__(tp_group)
        
        self.masked_attn = self_attn
        
        self.encoder_output = config.transformer_type == TransformerType.EncoderDecoder
        if self.encoder_output:
            self.cross_attn = cross_attn
            self.cross_attn_norm = cross_attn_norm
            
        self.masked_attn_norm = self_attn_norm
        self.logits_norm = logits_norm
        
        self.ffn = ffn
        
    @torch.no_grad()   
    def forward(self,
                x: Tensor,
                encoder_output: Optional[Tensor] = None,
                decoder_mask: Optional[Tensor] = None) -> Tensor:
        
        seq_len = x.size(dim=1)
        attn_out = self.masked_attn_norm(x, self.masked_attn(x, mask=decoder_mask))
        if self.encoder_output:
            attn_out = self.cross_attn_norm(
                attn_out, self.cross_attn(attn_out, encoder_output)
                )
        logits = self.logits_norm(attn_out, self.ffn(attn_out))
        return logits
    
class TPTransformerEncoderLayer(TPModule):
    def __init__(self,
                 config,
                 attn: TPModule,
                 attn_norm: Module,
                 ffn: TPModule,
                 logits_norm: Module,
                 tp_group: ProcessGroup) -> None:
        super().__init__(tp_group)
        
        self.attn = attn
            
        self.attn_norm = attn_norm
        self.logits_norm = logits_norm
        
        self.ffn = ffn
    @torch.no_grad()   
    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        attn_out = self.attn_norm(x, self.attn(x, mask))
        logits = self.logits_norm(attn_out, self.ffn(attn_out))
        return logits
        
        