import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .TPModuleGenerator import  TPModuleGenerator
from mytransformers.parallel.tensor_parallel.tp_layers import TPLayerNorm
from torch.nn import LayerNorm

    
class TPLayerNormGenerator(TPModuleGenerator):
    use_all_gather: bool = True
    @torch.no_grad()  
    def __new__(cls, module: LayerNorm, tp_group: ProcessGroup) -> TPLayerNorm:
        """create TPLayerNorm from torch.nn.LayerNorm"""
        if TPLayerNormGenerator.already_conferted(module): 
            return module
        
        tp_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)

        normalized_shape = module.normalized_shape[0]
        assert normalized_shape % tp_size == 0, "out_features must be divisible by tp_size"
        layer = TPLayerNorm(normalized_shape,
                            tp_group,
                            module.eps,
                            module.elementwise_affine,
                            cls.use_all_gather)
        if module.elementwise_affine:    
            weight = module.weight.chunk(tp_size, dim=0)[rank]
            bias = module.bias.chunk(tp_size, dim=0)[rank]
            layer.weight.copy_(weight.contiguous())
            layer.bias.copy_(bias.contiguous())
        device = torch.device(torch.cuda.current_device())  
        return layer.to(device)
        
        
        
        
        
        