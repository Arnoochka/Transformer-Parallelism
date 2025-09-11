from torch.nn import Module, ModuleList
import torch.nn as nn
from torch import Tensor
import torch
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

class MoELayer(Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.k = config.top_k
        
        self.gate = FeedForward(config)
        self.experts = ModuleList(
            [FeedForward(config)
             for _ in range(self.num_experts)])
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        gate_probs = self.gate(x_flat)
        topk_probs, topk_idx = torch.topk(gate_probs, k=self.k)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        token_idx = torch.arange(x_flat.size(0), device=x.device) \
            .unsqueeze(1).expand(-1, self.k).reshape(-1)
        expert_idx = topk_idx.reshape(-1)                                                                                      
        probs = topk_probs.reshape(-1)      
    
        logits_flat = torch.zeros_like(x_flat)
        for e in range(len(self.experts)):
            mask = (expert_idx == e)
            if mask.any():
                idx_e = token_idx[mask] 
                w_e   = probs[mask]     
                out_e = self.experts[e](x_flat[idx_e]) * w_e.unsqueeze(1)
                logits_flat.index_add_(0, idx_e, out_e)
    
        logits = logits_flat.view(batch_size, seq_len, hidden_size)
        return logits
        
class FFNType(Enum):
    MoE = MoELayer
    FFN = FeedForward
        