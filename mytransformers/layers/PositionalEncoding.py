import torch
from torch.nn import Module
from torch import Tensor
import numpy as np
    
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