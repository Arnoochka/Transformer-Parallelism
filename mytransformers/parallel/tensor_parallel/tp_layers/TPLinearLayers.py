import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .TPModule import TPModule
import torch.nn.functional as F

class TPLinear(TPModule):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 tp_group: ProcessGroup,
                 bias: bool = True):
        super().__init__(tp_group)

        self.register_buffer(
            "weight",
            torch.empty(out_features,
                        in_features,
                        requires_grad=False)
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty(out_features,
                            requires_grad=False)
            )
        else:
            self.bias = None

class TPColumnLinear(TPLinear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 tp_group: ProcessGroup,
                 bias: bool = True,
                 use_all_gather: bool = True):
        self.use_all_gather = use_all_gather
        tp_size = dist.get_world_size(tp_group)
        self.out_features_per_device = out_features // tp_size
        super().__init__(in_features, self.out_features_per_device, tp_group, bias=bias)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        W = [W_1, W_2, ..., W_n]
        X @ W = [X @ W_1, X @ W_2, ..., X @ W_n]
        """
        logits = F.linear(x, self.weight, self.bias)
        if self.use_all_gather:
            tp_size = dist.get_world_size(group=self.tp_group)
            logits_tensors = [torch.zeros_like(logits, device=logits.device) for _ in range(tp_size)]
            dist.all_gather(logits_tensors, logits, group=self.tp_group)
            logits = torch.cat(logits_tensors, dim=-1)

        return logits
    
class TPRowLinear(TPLinear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 tp_group: ProcessGroup,
                 bias: bool = True,
                 use_all_reduce: bool = True):
        self.use_all_reduce = use_all_reduce
        tp_size = dist.get_world_size(tp_group)
        self.in_features_per_device = in_features // tp_size
        super().__init__(self.in_features_per_device, out_features, tp_group, bias=bias)
    
    @torch.no_grad()  
    def forward(self, x: Tensor) -> Tensor:
        """
        X = [X_1, X_2, ..., X_n]
        W = [W_1, W_2, ..., W_n]
        X @ W = X_1 * W_1 + X_2 * W_2 + ... + X_n * W_n
        """
        logits = F.linear(x, self.weight, self.bias)
        if self.use_all_reduce:
            dist.all_reduce(logits, group=self.tp_group)
            
        return logits
        
        
        
        
        
        