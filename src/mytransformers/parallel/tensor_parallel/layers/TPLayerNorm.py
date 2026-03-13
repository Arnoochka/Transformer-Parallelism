import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .TPModule import TPModule


class TPLayerNorm(TPModule):
    """
    Базовый класс для тензорно-параллельной LayerNorm.

    Args:
        normalized_shape (int): Размер нормализуемой размерности (до разбиения).
        tp_group (ProcessGroup): Группа процессов для тензорного параллелизма.
        eps (float, optional): Малое число для численной стабильности. По умолчанию ``1e-5``.
        elementwise_affine (bool, optional): Если ``True``, используется обучаемая
            аффинная трансформация (веса и смещение). По умолчанию ``True``.
        use_all_gather (bool, optional): Если ``True``, собирает выходы всех устройств. По умолчанию ``True``.
    """
    def __init__(self,
                 normalized_shape: int,
                 tp_group: ProcessGroup,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 use_all_gather: bool = True):
        super().__init__(tp_group)
        tp_size = dist.get_world_size()
        self.global_normalized_shape = normalized_shape
        self.normalized_shape = normalized_shape // tp_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_all_gather = use_all_gather
        if self.elementwise_affine:
            self.register_buffer('weight',
                                 torch.empty(self.normalized_shape,
                                             requires_grad=False))
            self.register_buffer('bias',
                                 torch.empty(self.normalized_shape,
                                             requires_grad=False))
        else:
            self.register_buffer('weight', None)
            self.register_buffer('bias', None)


class TPSplittedLayerNorm(TPLayerNorm):
    """
    Тензорно-параллельная версия LayerNorm, где входы разделены между процессами.

    Каждый процесс получает часть признаков входа и вычисляет свои локальные статистики,
    после чего они суммируются с помощью ``all_reduce``.

    Args:
        normalized_shape (int): Размер нормализуемой размерности (до разбиения).
        tp_group (ProcessGroup): Группа процессов для тензорного параллелизма.
        eps (float, optional): Малое число для численной стабильности. По умолчанию ``1e-5``.
        elementwise_affine (bool, optional): Если ``True``, используется обучаемая
            аффинная трансформация (веса и смещение). По умолчанию ``True``.
        use_all_gather (bool, optional): Если ``True``, собирает выходы всех устройств. По умолчанию ``True``.
    """
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        input:
            X = [X_1, X_2, ..., X_n]
        output:
            Z = [Z_1, Z_2, ..., Z_n]
        """
        stats = torch.empty((*x.shape[:-1], 1, 2), device=x.device, dtype=x.dtype)

        stats[..., 0] = x.sum(dim=-1, keepdim=True)
        stats[..., 1] = (x * x).sum(dim=-1, keepdim=True)
        dist.all_reduce(stats, group=self.tp_group)
        
        stats = stats / self.global_normalized_shape
        mean = stats[..., 0]
        stats[..., 1].sub_(mean.square())
        var = stats[..., 1]
        
        normed = ((x - mean) / torch.sqrt(var + self.eps))
        if not self.use_all_gather:
            if self.elementwise_affine:
                return self.weight * normed + self.bias
            return normed
        else: 
            if self.elementwise_affine:
                logits_t = (self.weight * normed + self.bias).transpose(0, -1).contiguous()
            else:
                logits_t = normed
        tp_size = dist.get_world_size(group=self.tp_group)
        all_logits_t = torch.empty((logits_t.shape[0] * tp_size, *logits_t.shape[1:]),
                                   device=logits_t.device)
        dist.all_gather_into_tensor(all_logits_t, logits_t, group=self.tp_group)

        return all_logits_t.transpose(0, -1).contiguous()


class TPJoinedLayerNorm(TPLayerNorm):
    """
    Тензорно-параллельная версия LayerNorm, где вход объединён перед нормализацией,
    а затем разделяется обратно.

    Каждый процесс получает часть выходных признаков после общей нормализации.

    Args:
        normalized_shape (int): Размер нормализуемой размерности (до разбиения).
        tp_group (ProcessGroup): Группа процессов для тензорного параллелизма.
        eps (float, optional): Малое число для численной стабильности. По умолчанию ``1e-5``.
        elementwise_affine (bool, optional): Если ``True``, используется обучаемая
            аффинная трансформация (веса и смещение). По умолчанию ``True``.
        use_all_gather (bool, optional): Если ``True``, собирает выходы всех устройств. По умолчанию ``True``.
    """
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        input:
            X = [X, X, ... X]
        output:
            Z = [Z_1, Z_2, ..., Z_n]
        """
        stats = torch.empty((*x.shape[:-1], 1, 2), device=x.device, dtype=x.dtype)
        
        stats[..., 0] = x.sum(dim=-1, keepdim=True)
        stats[..., 1] = (x * x).sum(dim=-1, keepdim=True)
        
        stats = stats / self.global_normalized_shape
        mean = stats[..., 0]
        stats[..., 1].sub_(mean.square())
        var = stats[..., 1]
        rank = dist.get_rank(self.tp_group)
        world_size = dist.get_world_size(self.tp_group)
        print(f"{rank}, SHAPE:{x.shape}")
        x = torch.chunk(x, world_size, dim=-1)[rank]
        print(f"{rank}, SHAPE:{x.shape}")
        normed = ((x - mean) / torch.sqrt(var + self.eps))
        if not self.use_all_gather:
            if self.elementwise_affine:
                return self.weight * normed + self.bias
            return normed
        else: 
            if self.elementwise_affine:
                logits_t = self.weight * normed + self.bias
            else:
                logits_t = normed
            
        tp_size = dist.get_world_size(group=self.tp_group)
        all_logits_t = [torch.empty_like(logits_t) for _ in range(tp_size)]
        dist.all_gather(all_logits_t, logits_t, group=self.tp_group)
        return torch.cat(all_logits_t, dim=-1)
