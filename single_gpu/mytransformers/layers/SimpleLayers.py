import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F
import numpy as np


class FeedForward(Module):
    def __init__(self,
                 hidden_state: int,
                 ffn_dim: int,
                 func = F.relu,
                 dropout: float = 0.0,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.func = func
        
        self.fc1 = nn.Linear(hidden_state, ffn_dim, dtype=dtype)
        self.fc2 = nn.Linear(ffn_dim, hidden_state, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.func(self.fc1(x))
        logits = self.fc2(self.dropout(x))
        return logits
    
class AddNorm(Module):
    def __init__(self,
                 hidden_state: int,
                 eps: float = 0.00001,
                 elementwise_affine: bool = True,
                 bias: bool = True,
                 dropout: float = 0.0,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_state,
                                 eps,
                                 elementwise_affine,
                                 bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))
    
class PositionalEncoding(Module):
    def __init__(self, hidden_state, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_state)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_state, 2).float() * (-np.log(10000.0) / hidden_state))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        # x: (batch_size, seq_len, hidden_state)
        x = x + self.pe[:, :x.size(1), :]
        return x