from mytransformers.parallel.tensor_parallel.layers import TPLinear
from torch.distributed import ProcessGroup
import torch.distributed as dist
from torch.nn import Module
from .TPLinearLayersGenerator import TPColumnLinearGenerator, TPRowLinearGenerator
from .TPModuleGenerator import  TPModuleGenerator
from torch import Tensor
import torch

def get_chunks(weight: Tensor, num_chunks: int) -> list[Tensor]:
    return torch.split(weight, [weight.size(0) // num_chunks] * num_chunks, dim=0) 

class TPGroupedAttentionGenerator(TPModuleGenerator):
    out_gen = TPRowLinearGenerator
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        cls.out_gen.use_all_reduce = True
        module.out_proj = cls.out_gen(module.out_proj, tp_group)
        
        world_size = dist.get_world_size()
        module.num_query_heads = module.num_query_heads // world_size
        module.num_kv_heads = module.num_kv_heads // world_size
        
        return module
   
class TPSelfAttentionGenerator(TPModuleGenerator):
    query_key_value_gen = TPColumnLinearGenerator
    attention_gen = TPGroupedAttentionGenerator
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        cls.query_key_value_gen.use_all_gather = False
        world_size = dist.get_world_size(tp_group)
        assert module.num_query_heads % world_size == 0,\
            f"It is not possible to split query heads into {world_size} devices"
        assert module.num_kv_heads % world_size == 0,\
            f"It is not possible to split kv heads into {world_size} devices"
            
        num_query_heads_per_device = module.num_query_heads // world_size
        num_kv_heads_per_device = module.num_kv_heads // world_size
        query_size = module.num_query_heads * module.qk_dim
        key_size = module.num_kv_heads * module.qk_dim
        value_size = module.num_kv_heads * module.v_dim
        module.num_query_heads = num_query_heads_per_device
        module.num_kv_heads = num_kv_heads_per_device
        
        query_weight, key_weight, value_weight = torch.split(
            module.query_key_value.weight,
            [query_size, key_size, value_size], dim=0)
        rearrange_weight = cls.rearrange_heads(query_weight,
                                             key_weight,
                                             value_weight,
                                             world_size)
        module.query_key_value.weight.copy_(rearrange_weight)
        if module.query_key_value.bias is not None:
            query_bias, key_bias, value_bias = torch.split(
                module.query_key_value.bias,
                [query_size, key_size, value_size], dim=0)
            rearrange_bias = cls.rearrange_heads(query_bias,
                                                key_bias,
                                                value_bias,
                                                world_size)
            module.query_key_value.bias.copy_(rearrange_bias)
            
        module.query_key_value = cls.query_key_value_gen(module.query_key_value, tp_group)
        module.attention = cls.attention_gen(module.attention, tp_group)
        return module
    
    @staticmethod
    def rearrange_heads(query: Tensor,
                         key: Tensor,
                         value: Tensor,
                         num_chunks: int) -> TPLinear:
        chunks = []
        query_chunks = get_chunks(query, num_chunks)
        key_chunks = get_chunks(key, num_chunks)
        value_chunks = get_chunks(value, num_chunks)
        for query_chunk, key_chunk, value_chunk in zip(query_chunks,
                                                       key_chunks,
                                                       value_chunks):
            chunks.append(torch.cat([query_chunk, key_chunk, value_chunk]))
            
        return torch.cat(chunks)
        
    
class TPCrossAttentionGenerator(TPModuleGenerator):
    query_gen = TPColumnLinearGenerator
    key_value_gen = TPColumnLinearGenerator
    attention_gen = TPGroupedAttentionGenerator
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> Module:
        cls.key_value_gen.use_all_gather = False
        cls.query_gen.use_all_gather=  False
        world_size = dist.get_world_size(tp_group)
        assert module.num_query_heads % world_size == 0,\
            f"It is not possible to split query heads into {world_size} devices"
        assert module.num_kv_heads % world_size == 0,\
            f"It is not possible to split kv heads into {world_size} devices"
            
        num_query_heads_per_device = module.num_query_heads // world_size
        num_kv_heads_per_device = module.num_kv_heads // world_size
        key_size = module.num_kv_heads * module.qk_dim
        value_size = module.num_kv_heads * module.v_dim
        module.num_query_heads = num_query_heads_per_device
        module.num_kv_heads = num_kv_heads_per_device
        
        key_weight, value_weight = torch.split(
            module.key_value.weight,
            [key_size, value_size], dim=0)
        rearrange_weight = cls.rearrange_heads(key_weight,
                                               value_weight,
                                               world_size)
        module.key_value.weight.copy_(rearrange_weight)
        if module.key_value.bias is not None:
            key_bias, value_bias = torch.split(
                module.key_value.bias,
                [key_size, value_size], dim=0)
            rearrange_bias = cls.rearrange_heads(key_bias,
                                                 value_bias,
                                                 world_size)
            module.key_value.bias.copy_(rearrange_bias)
         
        module.query = cls.query_gen(module.query, tp_group) 
        module.key_value = cls.key_value_gen(module.key_value, tp_group)
        module.attention = cls.attention_gen(module.attention, tp_group)
        return module
    
    @staticmethod
    def rearrange_heads(key: Tensor,
                        value: Tensor,
                        num_chunks: int) -> TPLinear:
        chunks = []
        key_chunks = get_chunks(key, num_chunks)
        value_chunks = get_chunks(value, num_chunks)
        for key_chunk, value_chunk in zip(key_chunks,
                                                       value_chunks):
            chunks.append(torch.cat([key_chunk, value_chunk]))
            
        return torch.cat(chunks)
        
    
    
    
    

