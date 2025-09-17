import torch
from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .TPModuleGenerator import  TPModuleGenerator
from mytransformers.parallel.tensor_parallel.tp_layers import TPColumnLinear,TPRowLinear, TPModule
import warnings

    
class TPColumnLinearGenerator(TPModuleGenerator):
    use_all_gather: bool = True
    @torch.no_grad()  
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPColumnLinear:
        """create ColumnParallelLinear from torch.nn.Linear"""
        if isinstance(module, TPModule):
            warnings.warn(
                f"linear module is already converted in TPLinear: {type(module).__name__}",
                UserWarning,
                stacklevel=5)
            return module
        tp_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)

        in_features = module.in_features
        out_features = module.out_features
        add_bias = module.bias is not None
        assert out_features % tp_size == 0, "out_features must be divisible by tp_size"
        layer = TPColumnLinear(in_features,
                               out_features,
                               tp_group,
                               bias=add_bias,
                               use_all_gather=cls.use_all_gather)
            
        weight = module.weight.chunk(tp_size, dim=0)[rank]
        layer.weight.copy_(weight.contiguous())
        if add_bias:
            bias = module.bias.chunk(tp_size, dim=0)[rank]
            layer.bias.copy_(bias.contiguous())
        
        device = torch.device(torch.cuda.current_device())  
        return layer.to(device)
    
class TPRowLinearGenerator(TPModuleGenerator):
    use_all_reduce: bool = True
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPRowLinear:
        """create RowParallelLinear from torch.nn.Linear"""
        if isinstance(module, TPModule):
            warnings.warn(
                f"linear module is already converted in TPLinear: {type(module).__name__}",
                UserWarning,
                stacklevel=5)
            return module
        tp_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)

        in_features = module.in_features
        out_features = module.out_features
        add_bias = module.bias is not None
        assert in_features % tp_size == 0, "in_features must be divisible by tp_size"
        layer = TPRowLinear(in_features,
                            out_features,
                            tp_group,
                            bias=add_bias,
                            use_all_reduce=cls.use_all_reduce)
        
        weight = module.weight.chunk(tp_size, dim=1)[rank]
        layer.weight.copy_(weight.contiguous())
        if add_bias:
            layer.bias.copy_((module.bias / tp_size))

        device = torch.device(torch.cuda.current_device())
        return layer.to(device)
        
        
        
        
        
        