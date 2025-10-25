import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .TPModuleGenerator import  TPModuleGenerator
from mytransformers.parallel.tensor_parallel.layers import TPSplittedLayerNorm, TPJoinedLayerNorm
from torch.nn import LayerNorm

    
class TPSplittedLayerNormGenerator(TPModuleGenerator):
    use_all_gather: bool = True
    tp_group: ProcessGroup = None
    @torch.no_grad()  
    def __new__(cls, module: LayerNorm, device: torch.device) -> TPSplittedLayerNorm:
        """create TPLayerNorm from torch.nn.LayerNorm"""
        if TPSplittedLayerNormGenerator.already_converted(module): 
            return module
        
        tp_size = dist.get_world_size(cls.tp_group)
        rank = dist.get_rank(cls.tp_group)

        normalized_shape = module.normalized_shape[0]
        assert normalized_shape % tp_size == 0, "out_features must be divisible by tp_size"
        layer = TPSplittedLayerNorm(normalized_shape,
                                    cls.tp_group,
                                    module.eps,
                                    module.elementwise_affine,
                                    cls.use_all_gather)
        if module.elementwise_affine:    
            weight = module.weight.chunk(tp_size, dim=0)[rank]
            bias = module.bias.chunk(tp_size, dim=0)[rank]
            layer.weight.copy_(weight.contiguous())
            layer.bias.copy_(bias.contiguous())
        return layer.to(device)
    
    
class TPJoinedLayerNormGenerator(TPModuleGenerator):
    use_all_gather: bool = True
    tp_group: ProcessGroup = None
    @torch.no_grad()  
    def __new__(cls, module: LayerNorm, device: torch.device) -> TPJoinedLayerNorm:
        """create TPLayerNorm from torch.nn.LayerNorm"""
        if TPJoinedLayerNormGenerator.already_converted(module): 
            return module
        
        tp_size = dist.get_world_size(cls.tp_group)
        rank = dist.get_rank(cls.tp_group)

        normalized_shape = module.normalized_shape[0]
        assert normalized_shape % tp_size == 0, "out_features must be divisible by tp_size"
        layer = TPJoinedLayerNorm(normalized_shape,
                                  cls.tp_group,
                                  module.eps,
                                  module.elementwise_affine,
                                  cls.use_all_gather)
        
        if module.elementwise_affine:    
            weight = module.weight.chunk(tp_size, dim=0)[rank]
            bias = module.bias.chunk(tp_size, dim=0)[rank]
            layer.weight.copy_(weight.contiguous())
            layer.bias.copy_(bias.contiguous())
        return layer.to(device)
        
        
        
        
        
        