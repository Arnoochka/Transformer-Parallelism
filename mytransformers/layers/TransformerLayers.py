import torch
from torch.nn import Module
from torch import Tensor
from typing import Optional
from .SimpleLayers import AddNorm
import copy
        
        
class TransformerDecoderLayer(Module):
    def __init__(self,
                 hidden_state: int,
                 self_attn_model: Module,
                 ffn_model: Module,
                 encoder_output: bool = False,
                 cross_attn_model: Optional[Module] = None
                 ) -> None:
        super().__init__()
        
        self.masked_attn = copy.deepcopy(self_attn_model)
        
        self.encoder_output = encoder_output
        if encoder_output:
            self.cross_attn = copy.deepcopy(cross_attn_model)
            self.norm_attn_2 = AddNorm(hidden_state)
            
        self.norm_attn_1 = AddNorm(hidden_state)
        self.norm_logits = AddNorm(hidden_state)
        
        self.ffn = copy.deepcopy(ffn_model)
        
        
    def forward(self,
                x: Tensor,
                encoder_output: Optional[Tensor] = None,
                decoder_mask: Optional[Tensor] = None) -> Tensor:
        
        seq_len = x.size(dim=1)
        if decoder_mask is None:
            decoder_mask = torch.tril(torch.ones(seq_len, seq_len)) \
                .unsqueeze(0).unsqueeze(1).to(x.device)
        
        attn_out = self.norm_attn_1(x, self.masked_attn(x, mask=decoder_mask))
        if self.encoder_output:
            attn_out = self.norm_attn_2(
                attn_out, self.cross_attn(attn_out, encoder_output)
                )
            
        logits = self.norm_logits(attn_out, self.ffn(attn_out))
        return logits
    
class TransformerEncoderLayer(Module):
    def __init__(self,
                 hidden_state: int,
                 attn_model: Module,
                 ffn_model: Module) -> None:
        super().__init__()
        
        self.attn = copy.deepcopy(attn_model)
            
        self.norm_attn = AddNorm(hidden_state)
        self.norm_logits = AddNorm(hidden_state)
        
        self.ffn = copy.deepcopy(ffn_model)
        
    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        attn_out = self.norm_attn(x, self.attn(x, mask))
        logits = self.norm_logits(attn_out, self.ffn(attn_out))
        return logits