import torch
from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .TPModuleGenerator import  TPModuleGenerator
from mytransformers.parallel.tensor_parallel.tp_layers import (
    TPColumnLinear,
    TPRowLinear)

    
class TPColumnLinearGenerator(TPModuleGenerator):
    use_all_gather: bool = True
    @torch.no_grad()  
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPColumnLinear:
        """create ColumnParallelLinear from torch.nn.Linear"""
        tp_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)

        in_features = module.in_features
        out_features = module.out_features
        add_bias = module.bias is not None
        assert out_features % tp_size == 0, "out_features must be divisible by tp_size"
        
        layer = TPColumnLinear(in_features, out_features, tp_group,
                                     bias=add_bias, use_all_gather=cls.use_all_gather).to(torch.cuda.current_device())

        if rank == 0:
            device = torch.device(torch.cuda.current_device())
            w_chunks = [w.contiguous().to(device)
                        for w in module.weight.chunk(tp_size, dim=0)]
            b_chunks = ([b.contiguous().to(device)
                         for b in module.bias.chunk(tp_size, dim=0)]
                        if add_bias else None)
        else:
            w_chunks, b_chunks = None, None
        dist.scatter(layer.weight, scatter_list=w_chunks if rank == 0 else [], src=0, group=tp_group)
        if add_bias:
            dist.scatter(layer.bias, scatter_list=b_chunks if rank == 0 else [], src=0, group=tp_group)
            
        return layer
    
class TPRowLinearGenerator(TPModuleGenerator):
    use_all_reduce: bool = True
    @torch.no_grad()
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPRowLinear:
        """create RowParallelLinear from torch.nn.Linear"""
        tp_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)

        in_features = module.in_features
        out_features = module.out_features
        add_bias = module.bias is not None
        assert in_features % tp_size == 0, "in_features must be divisible by tp_size"

        layer = TPRowLinear(in_features, out_features, tp_group,
                                     bias=add_bias, use_all_reduce=cls.use_all_reduce).to(torch.cuda.current_device())

        if rank == 0:
            device = torch.device(torch.cuda.current_device())
            w_chunks = [w.contiguous().to(device)
                        for w in module.weight.chunk(tp_size, dim=1)]
            if add_bias:
                layer.bias = (module.bias / tp_size).to(device)
            else:
                layer.bias = None
        else:
            w_chunks = None
        dist.scatter(layer.weight, scatter_list=w_chunks if rank == 0 else [], src=0, group=tp_group)
        if add_bias:
            dist.broadcast(layer.bias, src=0, group=tp_group)

        return layer
        
        
        
        
        
        