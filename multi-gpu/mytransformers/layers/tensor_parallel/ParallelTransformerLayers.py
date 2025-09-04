from torch.distributed import ProcessGroup
import torch.distributed as dist
from torch.nn import Module
from torch.nn import Module
from .ParallelModule import TensorParallelModule, TensorParallelModuleGenerator
from ..SimpleLayers import AddNorm
from ..MoE import MoELayer
from ..TransformerLayers import TransformerType
from .ParallelAttention import ParallelCrossAttentionGenerator, ParallelSelfAttentionGenerator
from .ParallelLinearLayers import ColumnParallelLinearGenerator
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
    
    
class ParallelTransformerDecoderGenerator(TensorParallelModuleGenerator):
    config = None
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelTransformerDecoderLayer:
        ParallelSelfAttentionGenerator.config = cls.config
        ParallelCrossAttentionGenerator.config = cls.config
        
        use_encoder = cls.config.transformer_type == TransformerType.EncoderDecoder
        self_attn = ParallelSelfAttentionGenerator(module.self_attn, tp_group)
        ColumnParallelLinearGenerator.use_all_gather = True
        ffn = ColumnParallelLinearGenerator(module.ffn, tp_group)
        if use_encoder:
            cross_attn = ParallelCrossAttentionGenerator(module.cross_attn)
            return ParallelTransformerDecoderLayer(cls.config,
                                                   self_attn,
                                                   cross_attn,
                                                   ffn,
                                                   tp_group)
        else:
            return ParallelTransformerDecoderLayer(cls.config,
                                       self_attn,
                                       None,
                                       ffn,
                                       tp_group)
        
        
class ParallelTransformerEncoderGenerator(TensorParallelModuleGenerator):
    config = None
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelTransformerDecoderLayer:
        ParallelSelfAttentionGenerator.config = cls.config
        attn = ParallelSelfAttentionGenerator(module.self_attn, tp_group)
        ColumnParallelLinearGenerator.use_all_gather = True
        ffn = ColumnParallelLinearGenerator(module.ffn, tp_group)
        
        return ParallelTransformerEncoderLayer(cls.config,
                                               attn,
                                               ffn,
                                               tp_group)
        
        