import torch
import torch.nn as nn
from torch.nn import Module, ModuleList
from torch import Tensor
import copy
    
    
class TransformerEncoderDecoderModel(Module):
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
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_state)
        self.pos_encoding = copy.deepcopy(positional_encoding_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = ModuleList(
            [copy.deepcopy(encoder_layer)
             for _ in range(num_layers)]
        )
        self.decoder_layers = ModuleList(
            [copy.deepcopy(decoder_layer)
             for _ in range(num_layers)]
        )
            
        self.linear = nn.Linear(hidden_state, vocab_size, bias)
        
    def generate_mask(self, source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        
        source_mask = (source != self.pad_token_id).int().unsqueeze(1).unsqueeze(2)
        
        target_mask = (target != self.pad_token_id).int().unsqueeze(1).unsqueeze(2)
        seq_len = target.size(dim=1)
        casual_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=int)).to(target.device)
        target_mask = target_mask & casual_mask
        
        return source_mask, target_mask
    
    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        
        src_mask, tgt_mask = self.generate_mask(source, target)
        
        # Encoder
        output_encoder = self.dropout(
            self.pos_encoding(
                self.embedding(source)
                )
            )
        
        for layer in self.encoder_layers:
            output_encoder = layer(output_encoder, src_mask)
            
        # Decoder
        output_decoder = self.dropout(
            self.pos_encoding(
                self.embedding(target)
                )
            )
        
        for layer in self.decoder_layers:
            output_decoder = layer(output_decoder, output_encoder, tgt_mask, src_mask)
            
        logits = self.linear(output_decoder)
        return logits
    
class TransformerDecoderModel(Module):
    def __init__(self,
                 vocab_size: int,
                 pad_token_id: int,
                 num_layers: int,
                 hidden_state: int,
                 positional_encoding_model: Module,
                 decoder_model: Module,
                 bias: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_state)
        self.pos_encoding = copy.deepcopy(positional_encoding_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.decoder_layers = ModuleList(
            [copy.deepcopy(decoder_model)
             for _ in range(num_layers)]
        )
            
        self.linear = nn.Linear(hidden_state, vocab_size, bias)
        
    def generate_mask(self, target: Tensor) -> tuple[Tensor, Tensor]:
        
        target_mask = (target != self.pad_token_id).int().unsqueeze(1).unsqueeze(2)
        seq_len = target.size(dim=1)
        casual_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=int)).to(target.device)
        target_mask = target_mask & casual_mask
        
        return target_mask
    
    def forward(self, target: Tensor) -> Tensor:
        
        mask = self.generate_mask(target)
            
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
    
    
class TransformerEncoderModel(Module):
    def __init__(self,
                 vocab_size: int,
                 pad_token_id: int,
                 num_layers: int,
                 hidden_state: int,
                 positional_encoding_model: Module,
                 encoder_model: Module,
                 bias: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_state)
        self.pos_encoding = copy.deepcopy(positional_encoding_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = ModuleList(
            [copy.deepcopy(encoder_model)
             for _ in range(num_layers)]
        )
            
        self.linear = nn.Linear(hidden_state, vocab_size, bias)
        
    def create_padding_mask(self, source: Tensor) -> tuple[Tensor, Tensor]:
        
        source_mask = (source != self.pad_token_id).int().unsqueeze(1).unsqueeze(2)
        
        return source_mask
    
    def forward(self, source: Tensor) -> Tensor:
        
        mask = self.create_padding_mask(source)
        
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
    
        
        
        
        
        
        
        
        
            
        
            
        
        
        
        
        