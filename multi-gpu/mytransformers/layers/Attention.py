import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from torch.nn import Module
import copy

class GroupedQueryAttention(Module):
    def __init__(self,
                 hidden_state: int,
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
        
        self.out_proj = nn.Linear(self.num_query_heads * self.v_dim, hidden_state)
        
    def forward(self,
                Q: Tensor,
                K: Tensor,
                V: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        
        batch_size, seq_len, _ = Q.size()
        Q, K, V = self._reshape_query_key_value(Q, K, V)      
        attn_out = Q @ K.transpose(-2, -1) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_out = attn_out.masked_fill(mask==0, float('-inf'))
        attn_out: Tensor = self.dropout(
            self.softmax(attn_out)
        ) @ V
        attn_out = attn_out.view(batch_size, self.num_query_heads, seq_len, self.v_dim)
        logits = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.num_query_heads * self.v_dim)
        
        return self.out_proj(logits)
    
    def _reshape_query_key_value(self,
                                 Q: Tensor,
                                 K: Tensor,
                                 V: Tensor
                                 ) -> tuple[Tensor, Tensor, Tensor]:
        
        batch_size_q, seq_len_q, _ = Q.size()
        batch_size_k, seq_len_k, _ = K.size() 
        batch_size_v, seq_len_v, _ = V.size()  
        Q = Q.view(batch_size_q, seq_len_q, self.num_query_heads, self.qk_dim).transpose(1, 2)
        K = K.view(batch_size_k, seq_len_k, self.num_kv_heads, self.qk_dim).transpose(1, 2)
        V = V.view(batch_size_v, seq_len_v, self.num_kv_heads, self.v_dim).transpose(1, 2)
        
        Q_g = Q.view(batch_size_q, self.num_kv_heads, self.query_in_group, seq_len_q, self.qk_dim)
        K_g = K.unsqueeze(2)
        V_g = V.unsqueeze(2) 
        
        return Q_g, K_g, V_g
        
        
class AttentionKVCacheCore(Module):
    def __init__(self,
                 hidden_state: int,
                 num_query_heads: int,
                 num_kv_heads: int, 
                 qk_dim: int,
                 v_dim: int,
                 bias: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()
        
        self.hidden_state = hidden_state
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        
        self.attention = GroupedQueryAttention(
            hidden_state=hidden_state,
            qk_dim=qk_dim,
            v_dim=v_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout)
        
        self.use_kv_cache = False
        self.register_buffer('k_cache', None, persistent=True)
        self.register_buffer('v_cache', None, persistent=True)
    
    def enable_kv_cache(self) -> None:
        self.clear_kv_cache()
        self.use_kv_cache = True
        
    def disable_kv_cache(self) -> None:
        self.clear_kv_cache() 
        self.use_kv_cache = False   
    
    def clear_kv_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        
    def print_kv_cache_status(self) -> None:
        print(f"kv cache status:{self.use_kv_cache}")
        
        
    @staticmethod
    def call_kv_cache_method(model: Module, method_name: str) -> None:
        def call_method(module: Module) -> None:
            if hasattr(module, method_name):
                method = getattr(module, method_name)
                method()
        model.apply(call_method)

class SelfAttention(AttentionKVCacheCore):
    def __init__(self,
                 hidden_state: int,
                 num_query_heads: int,
                 num_kv_heads: int, 
                 qk_dim: int,
                 v_dim: int,
                 bias: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__(
            hidden_state=hidden_state,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            bias=bias,
            dropout=dropout)
        """
        Groupted-query self-attention 
        """
        
        self.query_key_value = nn.Linear(hidden_state,
                                         num_query_heads * qk_dim + num_kv_heads * (qk_dim + v_dim),
                                         bias=bias)
        
        
        
    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        Input:
            x: [batch_size, seq_len, hidden_state]
            encoder_output: [batch_size, seq_len, hidden_state]
            mask: [batch_size, 1, seq_len, seq_len]
        Output:
            logits: [batch_size, seq_len, hidden_state]
        """
        query_key_value = self.query_key_value(x)
        total_q_size = self.num_query_heads * self.qk_dim
        total_k_size = self.num_kv_heads * (self.qk_dim)
        Q: Tensor = query_key_value[..., :total_q_size]
        K: Tensor = query_key_value[..., total_q_size: total_q_size + total_k_size]
        V: Tensor = query_key_value[..., total_q_size + total_k_size:]
        
        if self.use_kv_cache:
            if (self.k_cache is None) or (self.v_cache is None):
                self.k_cache = K
                self.v_cache = V
            else:
               self.k_cache = torch.cat([self.k_cache, K], dim=1)
               self.v_cache = torch.cat([self.v_cache, V], dim=1) 
               
            K = self.k_cache
            V = self.v_cache
        
        return self.attention(Q, K, V, mask=mask)
    
class CrossAttention(AttentionKVCacheCore):
    def __init__(self,
                 hidden_state: int,
                 num_query_heads: int,
                 num_kv_heads: int, 
                 qk_dim: int,
                 v_dim: int,
                 bias: bool = False,
                 dropout: float = 0.0,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__(
            hidden_state=hidden_state,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            bias=bias,
            dropout=dropout)
        """
        Groupted-query cross-attention 
        """
        
        self.query = nn.Linear(hidden_state,
                               num_query_heads * qk_dim,
                               bias=bias,
                               dtype=dtype)
        
        self.key_value = nn.Linear(hidden_state,
                                   num_kv_heads * (qk_dim + v_dim))
        
        
    def forward(self,
                x: Tensor,
                encoder_output: Tensor) -> Tensor:
        """
        Input:
            x: [batch_size, seq_len, hidden_state]
            encoder_output: [batch_size, seq_len, hidden_state]
            mask: [seq_len, seq_len]
        Output:
            logits: [batch_size, seq_len, hidden_state]
        """
        
        Q: Tensor = self.query(x)
        if self.use_kv_cache:
            if (self.k_cache is None) or (self.v_cache is None):
                key_value = self.key_value(encoder_output)
                self.k_cache = key_value[..., :self.num_kv_heads * self.qk_dim]
                self.v_cache = key_value[..., self.num_kv_heads * self.qk_dim:]
            K: Tensor = self.k_cache
            V: Tensor = self.v_cache
        else:
            key_value = self.key_value(encoder_output)
            K = key_value[..., :self.num_kv_heads * self.qk_dim]
            V = key_value[..., self.num_kv_heads * self.qk_dim:]
            
        return self.attention(Q, K, V, mask=None)
    