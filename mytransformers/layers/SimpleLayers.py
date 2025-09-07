import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
import numpy as np

from enum import Enum



class FeedForward(Module):
    def __init__(self, config):
        super().__init__()
        self.func = nn.SELU()
        
        self.wi = nn.Linear(config.hidden_size, config.ffn_dim, bias=config.bias)
        self.wo = nn.Linear(config.ffn_dim, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor):
        x = self.func(self.wi(x))
        logits = self.wo(self.dropout(x))
        return logits
    
class AddNorm(Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size,
                                 config.eps,
                                 config.elementwise_affine)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))
    
class PositionalEncoding(Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(config.max_len, config.hidden_size)
        position = torch.arange(0, config.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.hidden_size, 2).float() * (-np.log(10000.0) / config.hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self,
                x: Tensor,
                pos_start: int = 0):
        # x: (batch_size, seq_len, hidden_size)
        x = x + self.pe[:, pos_start:pos_start + x.size(1), :]
        return x