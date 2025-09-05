from torch.distributed import ProcessGroup
import torch.distributed as dist
from torch.nn import Module
from torch.nn import Module
from .ParallelLinearLayersGenerator import ColumnParallelLinearGenerator, RowParallelLinearGenerator
from .ParallelModuleGenerator import  TensorParallelModuleGenerator
from .parallel_layers import ParallelAttention, ParallelCrossAttention, ParallelSelfAttention
import torch
import torch.nn as nn
from torch.nn import Linear
from torch import Tensor
from typing import Optional


def get_linear_from_weight(weight: Tensor,
                   hidden_size: int,
                   out_size: int,
                   bias: Optional[Tensor],
                   device: str | torch.device) -> Linear:
    
        proj = nn.Linear(hidden_size, out_size, bias=bias is not None, device=device)

        proj.weight.data.copy_(weight)
        
        if bias is not None:
            proj.bias.data.copy_(bias)
            
        return proj
        
class ParallelAttentionGenerator(TensorParallelModuleGenerator):
    config = None 
    attn: ParallelAttention = None
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelAttention:
        tp_size = dist.get_world_size(tp_group)
        assert cls.config.num_query_heads % tp_size == 0 and cls.config.num_kv_heads % tp_size == 0, \
            "num_query_heads and num_kv_heads must be divisible by tp_size"
            
        query_size = cls.config.num_query_heads * cls.config.qk_dim
        key_size = cls.config.num_kv_heads * cls.config.qk_dim
        value_size = cls.config.num_kv_heads * cls.config.v_dim
        hidden_size = cls.config.hidden_size
        device = module.query_weight.device
        
        query = get_linear_from_weight(module.query_weight,
                                       hidden_size,
                                       query_size,
                                       module.query_bias,
                                       device
                                       )
        key = get_linear_from_weight(module.key_weight,
                               hidden_size,
                               key_size,
                               module.key_bias,
                               device
                               )
        value = get_linear_from_weight(module.value_weight,
                               hidden_size,
                               value_size,
                               module.value_bias,
                               device
                               )
        out = module.attention.out_proj
        
        query_proj = ColumnParallelLinearGenerator(query, tp_group)
        key_proj = ColumnParallelLinearGenerator(key, tp_group)
        value_proj = ColumnParallelLinearGenerator(value, tp_group)
        out_proj = RowParallelLinearGenerator(out, tp_group)
        
        return cls.attn(cls.config,
                        query_proj,
                        key_proj,
                        value_proj,
                        out_proj,
                        tp_group)
    
   
class ParallelSelfAttentionGenerator(TensorParallelModuleGenerator):
    config = None
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelSelfAttention:
        
        ParallelAttentionGenerator.config = cls.config
        ParallelAttentionGenerator.attn = ParallelSelfAttention
        ColumnParallelLinearGenerator.use_all_gather = False
        RowParallelLinearGenerator.use_all_reduce = True
        
        query_size = cls.config.num_query_heads * cls.config.qk_dim
        key_size = cls.config.num_kv_heads * cls.config.qk_dim
        value_size = cls.config.num_kv_heads * cls.config.v_dim
        query_weight, key_weight, value_weight = torch.split(
            module.query_key_value.weight,
            [query_size, key_size, value_size], dim=0)
        query_bias, key_bias, value_bias = torch.split(
            module.query_key_value.bias,
            [query_size, key_size, value_size], dim=0)
        
        module.query_weight = query_weight
        module.key_weight = key_weight
        module.value_weight = value_weight
        module.query_bias = query_bias
        module.key_bias = key_bias
        module.value_bias = value_bias
        
        return ParallelAttentionGenerator(module, tp_group)
    
class ParallelCrossAttentionGenerator(TensorParallelModuleGenerator):
    config = None
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> ParallelCrossAttention:
        
        ParallelAttentionGenerator.config = cls.config
        ParallelAttentionGenerator.attn = ParallelCrossAttention
        ColumnParallelLinearGenerator.use_all_gather = False
        RowParallelLinearGenerator.use_all_reduce = True
        
        key_size = cls.config.num_kv_heads * cls.config.qk_dim
        value_size = cls.config.num_kv_heads * cls.config.v_dim
        query_weight = module.query.weight
        query_bias = module.query.bias
        key_weight, value_weight = torch.split(
            module.key_value.weight,
            [key_size, value_size])
        key_bias, value_bias = torch.split(
            module.key_value.bias,
            [key_size, value_size]
        )
        
        module.query_weight = query_weight
        module.key_weight = key_weight
        module.value_weight = value_weight
        module.query_bias = query_bias
        module.key_bias = key_bias
        module.value_bias = value_bias
        
        
        return ParallelAttentionGenerator(module, tp_group)
        
    
    
    
    

