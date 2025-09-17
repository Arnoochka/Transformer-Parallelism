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
        out_features_per_device = out_features // tp_size
        super().__init__(in_features, out_features_per_device, tp_group, bias=bias)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        W = [W_1, W_2, ..., W_n]
        X @ W = [X @ W_1, X @ W_2, ..., X @ W_n]
        """
        if not self.use_all_gather:
            return F.linear(x, self.weight, self.bias)
            
        logits_t = F.linear(x, self.weight, self.bias).transpose(0, 2).contiguous() 
        tp_size = dist.get_world_size(group=self.tp_group)
        all_logits_t = torch.empty((logits_t.shape[0] * tp_size, *logits_t.shape[1:]),
                                   device=logits_t.device)
        dist.all_gather_into_tensor(all_logits_t, logits_t, group=self.tp_group)

        return all_logits_t.transpose(0, 2).contiguous()
    
class TPRowLinear(TPLinear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 tp_group: ProcessGroup,
                 bias: bool = True,
                 use_all_reduce: bool = True):
        self.use_all_reduce = use_all_reduce
        tp_size = dist.get_world_size(tp_group)
        in_features_per_device = in_features // tp_size
        super().__init__(in_features_per_device, out_features, tp_group, bias=bias)
    
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
        
        
        
        
        
        