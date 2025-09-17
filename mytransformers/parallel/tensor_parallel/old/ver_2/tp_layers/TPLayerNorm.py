import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .TPModule import TPModule
import torch.nn.functional as F

class TPLayerNorm(TPModule):
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
        logits = F.linear(x, self.weight, self.bias)
        if self.use_all_gather:
            tp_size = dist.get_world_size(group=self.tp_group)
            logits_tensors = [torch.zeros_like(logits, device=logits.device) for _ in range(tp_size)]
            dist.all_gather(logits_tensors, logits, group=self.tp_group)
            logits = torch.cat(logits_tensors, dim=-1)

        return logits
        
        
        
        
        
        