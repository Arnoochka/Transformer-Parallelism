import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
import numpy as np
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .ParallelModule import TensorParallelModule

class ParallelLinear(TensorParallelModule):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 tp_group: ProcessGroup,
                 bias: bool = True):
        super().__init__(tp_group)

        self.register_buffer(
            "weight",
            torch.empty(in_features, out_features, device=torch.cuda.current_device())
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty(out_features, device=torch.cuda.current_device())
            )
        else:
            self.bias = None


class ColumnParallelLinear(ParallelLinear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 tp_group: ProcessGroup,
                 bias: bool = True,
                 use_all_gather: bool = True):
        self.use_all_gather = use_all_gather
        world_size = dist.get_world_size(tp_group)
        assert out_features % world_size == 0, "out_features must be divisible by world_size"
        self.out_features_per_device = out_features // world_size
        super().__init__(in_features, self.out_features_per_device, tp_group, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        W = [W_1, W_2, ..., W_n]
        X @ W = [X @ W_1, X @ W_2, ..., X @ W_n]
        """
        logits = x @ self.weight
        if self.bias is not None:
            logits = logits + self.bias

        if self.use_all_gather:
            world_size = dist.get_world_size(group=self.tp_group)
            logits_tensors = [torch.zeros_like(logits, device=logits.device) for _ in range(world_size)]
            dist.all_gather(logits_tensors, logits, group=self.tp_group)
            logits = torch.cat(logits_tensors, dim=-1)

        return logits

    @staticmethod
    def from_no_parallel(module: Module, tp_group, use_all_gather=True) -> TensorParallelModule:
        """create ColumnParallelLinear from torch.nn.Linear"""
        world_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)

        in_features = module.in_features
        out_features = module.out_features
        add_bias = module.bias is not None

        layer = ColumnParallelLinear(in_features, out_features, tp_group,
                                     bias=add_bias, use_all_gather=use_all_gather).to(torch.cuda.current_device())

        if rank == 0:
            w_chunks = list(module.weight.chunk(world_size, dim=1))
            b_chunks = list(module.bias.chunk(world_size, dim=0)) if add_bias else None
        else:
            w_chunks, b_chunks = None, None

        dist.scatter(layer.weight, scatter_list=w_chunks if rank == 0 else [], src=0, group=tp_group)

        if add_bias:
            dist.scatter(layer.bias, scatter_list=b_chunks if rank == 0 else [], src=0, group=tp_group)

        return layer
    
class RowParallelLinear(ParallelLinear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 tp_group: ProcessGroup,
                 bias: bool = True,
                 use_all_reduce: bool = True):
        self.use_all_reduce = use_all_reduce
        world_size = dist.get_world_size(tp_group)
        assert in_features % world_size == 0, "out_features must be divisible by world_size"
        self.in_features_per_device = in_features // world_size
        super().__init__(self.in_features_per_device, out_features, tp_group, bias=bias)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        X = [X_1, X_2, ..., X_n]
        W = [W_1, W_2, ..., W_n]
        X @ W = X_1 * W_1 + X_2 * W_2 + ... + X_n * W_n
        """
        
        logits = x @ self.weight
        if self.bias is not None:
            logits = logits + self.bias
        
        if self.use_all_reduce:
            dist.all_reduce(logits, group=self.tp_group)
            
        return logits
    
    @staticmethod
    def from_no_parallel(module: Module, tp_group, use_all_reduce=True) -> TensorParallelModule:
        """create RowParallelLinear from torch.nn.Linear"""
        world_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)

        in_features = module.in_features
        out_features = module.out_features
        add_bias = module.bias is not None

        layer = RowParallelLinear(in_features, out_features, tp_group,
                                     bias=add_bias, use_all_reduce=use_all_reduce).to(torch.cuda.current_device())

        if rank == 0:
            w_chunks = list(module.weight.chunk(world_size, dim=0))
            b_chunks = module.bias / world_size if add_bias else None
        else:
            w_chunks, b_chunks = None, None

        dist.scatter(layer.weight, scatter_list=w_chunks if rank == 0 else [], src=0, group=tp_group)

        if add_bias:
            dist.broadcast(b_chunks, src=0, group=tp_group)

        return layer
        
        
        
        
        
        