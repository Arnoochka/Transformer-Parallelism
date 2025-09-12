import torch
from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .TPModuleGenerator import  TPModuleGenerator
from mytransformers.parallel.tensor_parallel.tp_layers import TPColumnEmbedding, TPRowEmbedding, TPModule
import warnings

    
class TPColumnEmbeddingGenerator(TPModuleGenerator):
    use_all_gather: bool = True
    @torch.no_grad()  
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPColumnEmbedding:
        """create ColumnParallEmbedding from torch.nn.Embedding"""
        if isinstance(module, TPModule):
            warnings.warn(
                f"embedding module is already converted in TPLinear: {type(module).__name__}",
                UserWarning,
                stacklevel=5)
            return module
        tp_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)

        num_embeddings = module.num_embeddings
        embedding_dim = module.embedding_dim
        assert embedding_dim % tp_size == 0, "out_features must be divisible by tp_size"
        layer = TPColumnEmbedding(
            num_embeddings,
            embedding_dim,
            tp_group,
            module.padding_idx,
            module.max_norm,
            module.norm_type,
            module.sparse)
        weight = module.weight.chunk(tp_size, dim=1)[rank]
        layer.weight.copy_(weight.contiguous())
        
        device = torch.device(torch.cuda.current_device())  
        return layer.to(device)
    
class TPRowEmbeddingGenerator(TPModuleGenerator):
    use_all_reduce: bool = True
    @torch.no_grad()  
    def __new__(cls, module: Module, tp_group: ProcessGroup) -> TPColumnEmbedding:
        """create ColumnParallEmbedding from torch.nn.Embedding"""
        if isinstance(module, TPModule):
            warnings.warn(
                f"embedding module is already converted in TPLinear: {type(module).__name__}",
                UserWarning,
                stacklevel=5)
            return module
        tp_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)

        num_embeddings = module.num_embeddings
        embedding_dim = module.embedding_dim
        assert num_embeddings % tp_size == 0, "out_features must be divisible by tp_size"
        layer = TPRowEmbedding(
            num_embeddings,
            embedding_dim,
            tp_group,
            module.padding_idx,
            module.max_norm,
            module.norm_type,
            module.sparse)
        weight = module.weight.chunk(tp_size, dim=0)[rank]
        layer.weight.copy_(weight.contiguous())
        
        device = torch.device(torch.cuda.current_device())  
        return layer.to(device)