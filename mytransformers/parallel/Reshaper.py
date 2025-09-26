import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.nn import Module

class Reshaper(Module):
    def __init__(self,
                 module: Module,
                 group: ProcessGroup):
        super().__init__()
        self.group = group
        self.module = module
    @torch.no_grad()    
    def forward(self, *args, **kwargs) -> Tensor:
        return self.module.forward(*args, **kwargs)
        
class SimpleSplitter(Reshaper):
    def __init__(self,
                 module: Module,
                 dim: int,
                 group: ProcessGroup):
        super().__init__(module, group)
        self.dim = dim
        
    @torch.no_grad()
    def forward(self, *args, **kwargs):
        rank = dist.get_rank(self.group)
        world_size = dist.get_world_size(self.group)
        return super().forward(*args, **kwargs).chunk(world_size, self.dim)[rank]
    
class SimpleJoiner(Reshaper):
    def __init__(self,
                 module: Module,
                 dim: int,
                 group: ProcessGroup):
        super().__init__(module, group)
        self.dim = dim
        
    def forward(self, *args, **kwargs) -> Tensor:
        logits_t = super().forward(*args, **kwargs).transpose(0, self.dim).contiguous()
        tp_size = dist.get_world_size(group=self.group)
        all_logits_t = torch.empty((logits_t.shape[0] * tp_size, *logits_t.shape[1:]),
                                   device=logits_t.device)
        dist.all_gather_into_tensor(all_logits_t, logits_t, group=self.group)

        return all_logits_t.transpose(0, self.dim).contiguous()
    
class DimPartitionChanger(Reshaper):
    def __init__(self,
                 module: Module,
                 original_dim: int,
                 new_dim: int,
                 group: ProcessGroup):
        super().__init__(module, group)
        assert original_dim != new_dim, "It is pointless."
        self.original_dim = original_dim
        self.new_dim = new_dim
        
    def forward(self, *args, **kwargs) -> Tensor:
        rank = dist.get_rank(self.group)
        world_size = dist.get_world_size(self.group)
        logits_t = super().forward(*args, **kwargs).transpose(0, self.original_dim).contiguous()
        all_logits_t = torch.empty((logits_t.shape[0] * world_size, *logits_t.shape[1:]),
                                   device=logits_t.device)
        dist.all_gather_into_tensor(all_logits_t, logits_t, group=self.group)
        if self.new_dim == 0:
            logits_t = all_logits_t.chunk(world_size, self.original_dim)[rank] 
        else:
           logits_t = all_logits_t.chunk(world_size, self.new_dim)[rank]  
           
        return logits_t.transpose(0, self.original_dim).contiguous()
        
        
        
        
        
        
        
        
        