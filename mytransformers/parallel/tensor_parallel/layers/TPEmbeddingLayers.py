import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .TPModule import TPModule
import torch.nn.functional as F
from typing import Optional

class TPEmbedding(TPModule):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 tp_group: ProcessGroup,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 sparse: bool = False):
        super().__init__(tp_group)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.sparse = sparse
        self.register_buffer('weight',
                             torch.empty(
                                 num_embeddings,
                                 embedding_dim,
                                 requires_grad=False))
        
class TPColumnEmbedding(TPEmbedding):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 tp_group: ProcessGroup,
                 padding_idx: Optional[int]= None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 sparse: bool = False,
                 use_all_gather: bool = True):
        self.use_all_gather = use_all_gather
        tp_size = dist.get_world_size(tp_group)
        embedding_dim_per_device = embedding_dim // tp_size
        super().__init__(num_embeddings,
                         embedding_dim_per_device,
                         tp_group,
                         padding_idx,
                         max_norm,
                         norm_type,
                         sparse)
        
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        input: X
        output = [Z_1, Z_2, ..., Z_n]
        """
        if not self.use_all_gather:
            return F.embedding(x,
                                 self.weight,
                                 padding_idx=self.padding_idx,
                                 max_norm=self.max_norm,
                                 norm_type=self.norm_type,
                                 sparse=self.sparse)
        logits_t =  F.embedding(x,
                                self.weight,
                                padding_idx=self.padding_idx,
                                max_norm=self.max_norm,
                                norm_type=self.norm_type,
                                sparse=self.sparse).transpose(0, 2).contiguous()
        
        tp_size = dist.get_world_size(group=self.tp_group)
        all_logits_t = torch.empty((logits_t.shape[0] * tp_size, *logits_t.shape[1:]),
                                   device=logits_t.device)
        dist.all_gather_into_tensor(all_logits_t, logits_t, group=self.tp_group)

        return all_logits_t.transpose(0, 2).contiguous()
    
class TPRowEmbedding(TPEmbedding):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 tp_group: ProcessGroup,
                 padding_idx: Optional[int]= None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 sparse: bool = False,
                 use_all_reduce: bool = True):
        self.use_all_reduce = use_all_reduce
        tp_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)
        num_embeddings_per_device = num_embeddings // tp_size
        self.start_idx = rank * num_embeddings_per_device
        self.end_idx = (rank + 1) * num_embeddings_per_device
        self.max_idx = num_embeddings_per_device - 1
        super().__init__(num_embeddings_per_device,
                         embedding_dim,
                         tp_group,
                         padding_idx,
                         max_norm,
                         norm_type,
                         sparse)
    
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        input: X
        output = Z_1 + Z_2 + ... +Z_n
        """
        mask = ((x >= self.start_idx) & (x < self.end_idx)).unsqueeze(-1).float()
        x_local = (x - self.start_idx).clamp(min=0, max=self.max_idx)
        logits = F.embedding(x_local,
                             self.weight,
                             padding_idx=self.padding_idx,
                             max_norm=self.max_norm,
                             norm_type=self.norm_type,
                             sparse=self.sparse)

        logits = logits * mask
        if self.use_all_reduce:
            dist.all_reduce(logits, group=self.tp_group)

        return logits
        
        
        