import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
import numpy as np
from .ParallelLinearLayers import ColumnParallelLinear, RowParallelLinear, ParallelLinear
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .ParallelModule import TensorParallelModule



class ParallelFeedForward(TensorParallelModule):
    def __init__(self,
                 config,
                 wi: Module,
                 wo: Module,
                 tp_group: ProcessGroup):
        super().__init__(tp_group)
        self.func = nn.ReLU()
        self.wi = ColumnParallelLinear.from_no_parallel(wi,
                                                        tp_group,
                                                        use_all_gather=False)
        self.wo = RowParallelLinear.from_no_parallel(wo,
                                                     tp_group=self.tp_group,
                                                     use_all_reduce=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.func(self.wi(x))
        logits = self.wo(self.dropout(x))
        return logits
    
    @staticmethod
    def from_no_parallel(module: Module, tp_group: ProcessGroup, config):
        return ParallelFeedForward(config, module.wi, module.wo, tp_group)
        
    
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