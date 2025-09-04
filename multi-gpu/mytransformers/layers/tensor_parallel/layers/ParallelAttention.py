from torch.distributed import ProcessGroup
import torch.distributed as dist
from torch.nn import Module
from torch.nn import Module
from .ParallelLinearLayers import ColumnParallelLinearGenerator, RowParallelLinearGenerator
from .ParallelModule import TensorParallelModule, TensorParallelModuleGenerator
import torch
import torch.nn as nn
from torch.nn import Linear
from torch import Tensor
from typing import Optional

class ParallelGroupedQueryAttention(TensorParallelModule):
    def __init__(self,
                 config,
                 out_proj: TensorParallelModule,
                 tp_group: ProcessGroup):
        super().__init__(tp_group)
        
        self.qk_dim = config.qk_dim
        self.v_dim = config.v_dim
        self.num_query_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        assert self.num_query_heads % self.num_kv_heads == 0, \
            "num_query_heads must be divisible by num_kv_heads"
        self.query_in_group = self.num_query_heads // self.num_kv_heads
        
        tp_size = dist.get_world_size(tp_group)
        self.num_query_heads_per_device = self.num_query_heads // tp_size
        self.num_kv_heads_per_device = self.num_kv_heads // tp_size
        
        self.scale = 1.0 / (self.qk_dim ** 0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = out_proj
        
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
        Q = Q.view(batch_size_q, seq_len_q, self.num_query_heads_per_device, self.qk_dim).transpose(1, 2)
        K = K.view(batch_size_k, seq_len_k, self.num_kv_heads_per_device, self.qk_dim).transpose(1, 2)
        V = V.view(batch_size_v, seq_len_v, self.num_kv_heads_per_device, self.v_dim).transpose(1, 2)
        
        Q_g = Q.view(batch_size_q, self.num_kv_heads_per_device, self.query_in_group, seq_len_q, self.qk_dim)
        K_g = K.unsqueeze(2)
        V_g = V.unsqueeze(2) 
        
        return Q_g, K_g, V_g
        
        
class ParallelAttentionKVCacheCore(TensorParallelModule):
    def __init__(self,
                 config,
                 out_proj: TensorParallelModule,
                 tp_group: ProcessGroup) -> None:
        super().__init__(tp_group)
        
        self.hidden_state = config.hidden_state
        self.num_query_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        self.qk_dim = config.qk_dim
        self.v_dim = config.v_dim
        
        self.attention = ParallelGroupedQueryAttention(config,
                                                       out_proj,
                                                       tp_group)
        
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
        
class ParallelAttention(ParallelAttentionKVCacheCore):
    def __init__(self,
                 config,
                 query: TensorParallelModule,
                 key: TensorParallelModule,
                 value: TensorParallelModule,
                 out_proj: TensorParallelModule,
                 tp_group: ProcessGroup) -> None:
        super().__init__(config, out_proj, tp_group)
        """
        Groupted-query self-attention 
        """
        self.query = query
        self.key = key
        self.value = value
        
    def get_key_value(self, x: Tensor) -> tuple[Tensor, Tensor]:
        K = self.key(x)
        V = self.value(x)
        return K, V

class ParallelSelfAttention(ParallelAttention):     
        
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
        Q = self.query(x)
        K, V = self.get_key_value(x)
        
        return self.attention(Q, K, V, mask=mask)
    
    def get_key_value(self, x: Tensor) -> tuple[Tensor, Tensor]:
        K, V = super().get_key_value(x)
        
        if self.use_kv_cache:
            if (self.k_cache is None) or (self.v_cache is None):
                self.k_cache = K
                self.v_cache = V
            else:
               self.k_cache = torch.cat([self.k_cache, K], dim=1)
               self.v_cache = torch.cat([self.v_cache, V], dim=1) 
               
            K = self.k_cache
            V = self.v_cache
        
        return K, V
        
    
class ParallelCrossAttention(ParallelAttention):
        
        
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
        K, V = self.get_key_value(encoder_output)
        
        return self.attention(Q, K, V, mask=None)
    
    def get_key_value(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.use_kv_cache:
            if (self.k_cache is None) or (self.v_cache is None):
                K, V = super().get_key_value(x)
                self.k_cache, self.v_cache = K, V
            K: Tensor = self.k_cache
            V: Tensor = self.v_cache
        else:
            K, V = super().get_key_value(x)
        
        return K, V
        
    
    
    
    

