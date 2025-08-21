import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import numpy as np
from torch.nn import Module

class GroupedQueryAttention(Module):
    def __init__(self,
                 qk_dim: int,
                 v_dim: int,
                 num_query_heads: int,
                 num_kv_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        assert num_query_heads % num_kv_heads == 0, \
            "num_query_heads must be divisible by num_kv_heads"
        self.query_in_group = num_query_heads // num_kv_heads
        
        self.scale = 1.0 / (qk_dim ** 0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                Q: Tensor, K: Tensor, V: Tensor,
                mask: Optional[Tensor] = None
                ) -> Tensor:
        
        batch_size, seq_len, _ = Q.size() 
        Q = Q.view(batch_size, seq_len, self.num_query_heads, self.qk_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.qk_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.v_dim).transpose(1, 2)
        
        Q_g = Q.view(batch_size, self.num_kv_heads, self.query_in_group, seq_len, self.qk_dim)
        K_g = K.unsqueeze(2)
        V_g = V.unsqueeze(2) 
        
        attn_out = Q_g @ K_g.transpose(-2, -1) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_out = attn_out.masked_fill(mask==0, float('-inf'))
            
        attn_out: Tensor = self.dropout(
            self.softmax(attn_out)
        ) @ V_g
        
        return attn_out

    


class SelfAttention(Module):
    def __init__(self,
                 hidden_state: int,
                 num_query_heads: int,
                 num_kv_heads: int, 
                 qk_dim: int,
                 v_dim: int,
                 bias: bool = False,
                 dropout: float = 0.0,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        
        """
        Groupted-query self-attention 
        """
        
        assert num_query_heads % num_kv_heads == 0, \
            "num_query_heads must be divisible by num_kv_heads"
        self.query_in_group = num_query_heads // num_kv_heads
        
        self.query_key_value = nn.Linear(hidden_state,
                                         num_query_heads * qk_dim + num_kv_heads * (qk_dim + v_dim),
                                         bias=bias,
                                         dtype=dtype)
        
        self.qk_dim = qk_dim
        self.scale = 1.0 / (qk_dim ** 0.5)
        self.v_dim = v_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        
        self.out_proj = nn.Linear(self.num_query_heads * self.v_dim, hidden_state, dtype=dtype)
        self.attention = GroupedQueryAttention(qk_dim,
                                               v_dim,
                                               num_query_heads,
                                               num_kv_heads,
                                               dropout)
        
    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None
                ) -> Tensor:
        """
        Input:
            x: [batch_size, seq_len, hidden_state]
            encoder_output: [batch_size, seq_len, hidden_state]
            mask: [batch_size, 1, seq_len, seq_len]
        Output:
            logits: [batch_size, seq_len, hidden_state]
        """
        
        # [hidden_states, num_query_heads * qk_dim + num_kv_heads * (qk_dim + v_dim)]
        query_key_value = self.query_key_value(x)
        total_q_size = self.num_query_heads * self.qk_dim
        total_k_size = self.num_kv_heads * (self.qk_dim)
        Q: Tensor = query_key_value[..., :total_q_size]
        K: Tensor = query_key_value[..., total_q_size: total_q_size + total_k_size]
        V: Tensor = query_key_value[..., total_q_size + total_k_size:]
        
        batch_size, seq_len, hidden_state = x.size()
        attn_out: Tensor = self.attention(Q, K, V, mask=mask)
        
        logits = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.num_query_heads * self.v_dim)
        
        # [batch_size, seq_len, dim]
        
        return self.out_proj(logits)
    
class CrossAttention(Module):
    def __init__(self,
                 hidden_state: int,
                 num_query_heads: int,
                 num_kv_heads: int, 
                 qk_dim: int,
                 v_dim: int,
                 bias: bool = False,
                 dropout: float = 0.0,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        
        """
        Groupted-query cross-attention 
        """
        
        assert num_query_heads % num_kv_heads == 0, \
            "num_query_heads must be divisible by num_kv_heads"
        self.query_in_group = num_query_heads // num_kv_heads
        
        self.query = nn.Linear(hidden_state,
                               num_query_heads * qk_dim,
                               bias=bias,
                               dtype=dtype)
        
        self.key_value = nn.Linear(hidden_state,
                                   num_kv_heads * (qk_dim + v_dim))
        
        self.qk_dim = qk_dim
        self.scale = 1.0 / (qk_dim ** 0.5)
        self.v_dim = v_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        
        self.out_proj = nn.Linear(self.num_query_heads * self.v_dim, hidden_state, dtype=dtype)
        
        self.attention = GroupedQueryAttention(qk_dim,
                                               v_dim,
                                               num_query_heads,
                                               num_kv_heads,
                                               dropout)
        
        
    def forward(self,
                x: Tensor,
                encoder_output: Tensor,
                mask: Optional[Tensor] = None
                ) -> Tensor:
        """
        Input:
            x: [batch_size, seq_len, hidden_state]
            encoder_output: [batch_size, seq_len, hidden_state]
            mask: [seq_len, seq_len]
        Output:
            logits: [batch_size, seq_len, hidden_state]
        """
        
        Q: Tensor = self.query(x)
        
        key_value = self.key_value(encoder_output)
        
        K: Tensor = key_value[..., :self.num_kv_heads * self.qk_dim]
        V: Tensor = key_value[..., self.num_kv_heads * self.qk_dim:]
        
        batch_size, seq_len, hidden_state = x.size()
        
        attn_out: Tensor = self.attention(Q, K, V, mask=mask)
        
        logits = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.num_query_heads * self.v_dim)
        
        # [batch_size, seq_len, dim]
        
        return self.out_proj(logits)
    
    