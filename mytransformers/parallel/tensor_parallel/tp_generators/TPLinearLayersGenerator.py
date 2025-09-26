import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .TPModuleGenerator import  TPModuleGenerator
from mytransformers.parallel.tensor_parallel.tp_layers import TPColumnLinear, TPRowLinear
from torch.nn import Linear

    
class TPColumnLinearGenerator(TPModuleGenerator):
    use_all_gather: bool = True
    tp_group: ProcessGroup = None
    @torch.no_grad()  
    def __new__(cls, module: Linear, device: torch.device) -> TPColumnLinear:
        """create ColumnParallelLinear from torch.nn.Linear"""
        if TPColumnLinearGenerator.already_converted(module): 
            return module
        tp_size = dist.get_world_size(cls.tp_group)
        rank = dist.get_rank(cls.tp_group)

        in_features = module.in_features
        out_features = module.out_features
        add_bias = module.bias is not None
        assert out_features % tp_size == 0, "out_features must be divisible by tp_size"
        layer = TPColumnLinear(in_features,
                               out_features,
                               cls.tp_group,
                               bias=add_bias,
                               use_all_gather=cls.use_all_gather)
            
        weight = module.weight.chunk(tp_size, dim=0)[rank]
        layer.weight.copy_(weight.contiguous())
        if add_bias:
            bias = module.bias.chunk(tp_size, dim=0)[rank]
            layer.bias.copy_(bias.contiguous())
        
        return layer.to(device)
    
class TPRowLinearGenerator(TPModuleGenerator):
    use_all_reduce: bool = True
    tp_group: ProcessGroup = None
    @torch.no_grad()
    def __new__(cls, module: Linear, device: torch.device) -> TPRowLinear:
        """create RowParallelLinear from torch.nn.Linear"""
        if TPRowLinearGenerator.already_converted(module):
            return module
        tp_size = dist.get_world_size(cls.tp_group)
        rank = dist.get_rank(cls.tp_group)

        in_features = module.in_features
        out_features = module.out_features
        add_bias = module.bias is not None
        assert in_features % tp_size == 0, "in_features must be divisible by tp_size"
        layer = TPRowLinear(in_features,
                            out_features,
                            cls.tp_group,
                            bias=add_bias,
                            use_all_reduce=cls.use_all_reduce)
        
        weight = module.weight.chunk(tp_size, dim=1)[rank]
        layer.weight.copy_(weight.contiguous())
        if add_bias:
            layer.bias.copy_((module.bias / tp_size))

        return layer.to(device)
        
        
        
        
        
        