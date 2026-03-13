import torch.nn as nn
from torch.nn import Module
    
class AddNorm(Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size,
                                 config.eps,
                                 config.elementwise_affine)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))