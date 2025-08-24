import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch.nn import ModuleList

class FeedForward(Module):
    def __init__(self, config):
        super().__init__()
        self.func = nn.ReLU()
        
        self.fc1 = nn.Linear(config.hidden_state, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.hidden_state)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.func(self.fc1(x))
        logits = self.fc2(self.dropout(x))
        return logits
    
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
    
class PositionalEncoding(Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(config.max_len, config.hidden_state)
        position = torch.arange(0, config.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.hidden_state, 2).float() * (-np.log(10000.0) / config.hidden_state))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self,
                x: Tensor,
                pos_start: int = 0):
        # x: (batch_size, seq_len, hidden_state)
        x = x + self.pe[:, pos_start:pos_start + x.size(1), :]
        return x