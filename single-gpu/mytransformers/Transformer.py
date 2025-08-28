import torch
import torch.nn as nn
from torch.nn import Module, ModuleList
from .layers import AttentionKVCacheCore
from torch import Tensor
from typing import Optional

import copy

class TransformerCore(Module):
    def __init__(self,
                 vocab_size: int,
                 pad_token_id: int,
                 num_layers: int,
                 hidden_state: int,
                 positional_encoding_model: Module,
                 bias: bool = False,
                 dropout: float = 0.0
                 ) -> None:
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_state)
        self.pos_encoding = copy.deepcopy(positional_encoding_model)
        
        self.dropout = nn.Dropout(dropout)
            
        self.linear = nn.Linear(hidden_state, vocab_size, bias)
        
    def generate(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def _generate_source_mask(self, source: Tensor) -> Tensor:
        return (source != self.pad_token_id).int().unsqueeze(1).unsqueeze(2)
    
    def _generate_target_mask(self, target: Tensor) -> Tensor:
        target_mask = (target != self.pad_token_id).int().unsqueeze(1).unsqueeze(2)
        seq_len = target.size(dim=1)
        casual_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=int)).to(target.device)
        target_mask = target_mask & casual_mask
        
        return target_mask

    
class TransformerEncoderDecoderModel(TransformerCore):
    def __init__(self,
                 vocab_size: int,
                 pad_token_id: int,
                 num_layers: int,
                 hidden_state: int,
                 positional_encoding_model: Module,
                 encoder_layer: Module,
                 decoder_layer: Module,
                 bias: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            num_layers=num_layers,
            hidden_state=hidden_state,
            positional_encoding_model=positional_encoding_model,
            bias=bias,
            dropout=dropout)
        
        self.encoder_layers = ModuleList(
            [copy.deepcopy(encoder_layer)
             for _ in range(num_layers)]
        )
        self.decoder_layers = ModuleList(
            [copy.deepcopy(decoder_layer)
             for _ in range(num_layers)]
        )
    
    def __get_output_encoder(self, source: Tensor, src_mask: Tensor) -> Tensor:
        output_encoder = self.dropout(
            self.pos_encoding(
                self.embedding(source)
                )
            )
        
        for layer in self.encoder_layers:
            output_encoder = layer(output_encoder, src_mask)
            
        return output_encoder
    
    def __get_output_decoder(self,
                             target: Tensor,
                             tgt_mask: Tensor,
                             output_encoder: Tensor,
                             pos_seq: int = 0) -> Tensor:
        
        output_decoder = self.dropout(
            self.pos_encoding(
                self.embedding(target),
                pos_start = pos_seq))
        
        for layer in self.decoder_layers:
            output_decoder = layer(output_decoder,
                                   output_encoder,
                                   tgt_mask)
            
        return output_decoder
        
    
    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        
        src_mask = self._generate_source_mask(source)
        tgt_mask = self._generate_target_mask(target)
        
        output_encoder = self.__get_output_encoder(source, src_mask)
            
        output_decoder = self.__get_output_decoder(target, tgt_mask, output_encoder)
            
        logits = self.linear(output_decoder)
        return logits
    
    @torch.no_grad
    def generate(self, context: list,
                 tokens: list,
                 stop_token: int,
                 max_len: int,
                 device: str = 'cuda',) -> list:
        
        current_len = len(tokens)
        AttentionKVCacheCore.call_kv_cache_method(self, "enable_kv_cache")
        
        source = torch.tensor([context]).to(device)
        src_mask = self._generate_source_mask(source)
        output_encoder = self.__get_output_encoder(source, src_mask)
        
        target = torch.tensor([tokens]).to(device)
        tgt_mask = self._generate_target_mask(target)
        next_token = self.__get_next_token(target,
                                           tgt_mask,
                                           0,
                                           output_encoder)
        current_len += 1
        tokens.append(next_token)
        
        while current_len < max_len and next_token != stop_token:
            target = torch.tensor([[next_token]]).to(device)
            next_token = self.__get_next_token(target,
                                               None,
                                               current_len - 1,
                                               output_encoder)
            current_len += 1
            tokens.append(next_token)
            
        AttentionKVCacheCore.call_kv_cache_method(self, "clear_kv_cache")  
        return tokens
    
    def __get_next_token(self,
                         target: Tensor,
                         tgt_mask: Tensor,
                         pos_seq: int,
                         output_encoder: Tensor) -> int:
        
        output_decoder = self.__get_output_decoder(target,
                                                   tgt_mask,
                                                   output_encoder,
                                                   pos_seq=pos_seq)
        
        logits = self.linear(output_decoder)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        
        return next_token
    
class TransformerDecoderModel(TransformerCore):
    def __init__(self,
                 vocab_size: int,
                 pad_token_id: int,
                 num_layers: int,
                 hidden_state: int,
                 positional_encoding_model: Module,
                 decoder_model: Module,
                 bias: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            num_layers=num_layers,
            hidden_state=hidden_state,
            positional_encoding_model=positional_encoding_model,
            bias=bias,
            dropout=dropout)
        
        self.decoder_layers = ModuleList(
            [copy.deepcopy(decoder_model)
             for _ in range(num_layers)]
        )
    
    def forward(self, target: Tensor) -> Tensor:
        
        mask = self._generate_target_mask(target)
            
        # Decoder
        output_decoder = self.dropout(
            self.pos_encoding(
                self.embedding(target)
                )
            )
        
        for layer in self.decoder_layers:
            output_decoder = layer(output_decoder, decoder_mask=mask)
            
        logits = self.linear(output_decoder)
        return logits
    
    def generate(self,
                 tokens: list,
                 stop_token: int,
                 max_len: int,
                 device: str = 'cuda') -> Tensor:
        
        current_len = 0
        
        with torch.no_grad:
            next_token = self.__get_next_token(tokens, device=device)
            tokens.append(next_token)
            current_len += 1
            
            while current_len < max_len & next_token != stop_token:
                next_token = self.__get_next_token([next_token], device=device)
                tokens.append(next_token)
                
                current_len += 1

        return tokens
    
    def __get_next_token(self, tokens: list, device: str = 'cuda') -> int:
        x = torch.tensor([tokens]).to(device)
        logits = self.forward(x)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).item()  
        
        return next_token  
    
    
class TransformerEncoderModel(TransformerCore):
    def __init__(self,
                 vocab_size: int,
                 pad_token_id: int,
                 num_layers: int,
                 hidden_state: int,
                 positional_encoding_model: Module,
                 encoder_model: Module,
                 bias: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            num_layers=num_layers,
            hidden_state=hidden_state,
            positional_encoding_model=positional_encoding_model,
            bias=bias,
            dropout=dropout)
        
        self.encoder_layers = ModuleList(
            [copy.deepcopy(encoder_model)
             for _ in range(num_layers)]
        )
    
    def forward(self, source: Tensor) -> Tensor:
        
        mask = self._generate_source_mask(source)
        
        # Encoder
        output_encoder = self.dropout(
            self.pos_encoding(
                self.embedding(source)
                )
            )
        
        for layer in self.encoder_layers:
            output_encoder = layer(output_encoder, mask)
            
        logits = self.linear(output_encoder)
        return logits