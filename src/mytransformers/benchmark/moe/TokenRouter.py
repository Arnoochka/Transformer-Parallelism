from typing import Tuple
import torch
from torch import Tensor
from enum import Enum

class _TokenRouter:
    
    @staticmethod
    def uniform(router_logits: Tensor, top_k: int) -> Tuple[Tensor, Tensor]:
        """
        равномерное распределение
        """
        num_experts = router_logits.shape[-1]
        routing_weights = torch.nn.functional.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_index = torch.topk(routing_weights, top_k, dim=-1)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        top_k_index = torch.randint(
            low=0,
            high=num_experts,
            size=top_k_index.size(),
            device=router_logits.device
        )
        return top_k_index, top_k_weights.to(router_logits.dtype)
    
    @staticmethod
    def weighted(router_logits: Tensor, top_k: int) -> Tuple[Tensor, Tensor]:
        """
        распределение по заданным весам
        """
        p = _TokenRouter.weighted.weights.float()
        p = p / p.sum()

        batch = router_logits.size(0)
        idx = torch.multinomial(p.expand(batch, -1), top_k, replacement=False)
        w = p.expand(batch, -1).gather(-1, idx)
        w = w / w.sum(dim=-1, keepdim=True)
        return idx, w

    @staticmethod
    def multimodal(
        router_logits: Tensor,
        top_k: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Дисбалансное распределение с несклоькими "горячими" кластерами
        """
        num_experts = router_logits.size(-1)
        x = torch.arange(num_experts, dtype=torch.float32, device=router_logits.device)
        p = sum(torch.exp(-0.5 * ((x - c) / _TokenRouter.multimodal.std) ** 2) for c in _TokenRouter.multimodal.centers)
        p = p / p.sum()

        batch = router_logits.size(0)
        idx = torch.multinomial(p.expand(batch, -1), top_k, replacement=False)
        w = p.expand(batch, -1).gather(-1, idx)
        w = w / w.sum(dim=-1, keepdim=True)
        return idx, w.to(router_logits.dtype)

    @staticmethod
    def zipf(
        router_logits: Tensor,
        top_k: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Высокий дисбаланс с монотонным спадом — expert collapse.
        """
        num_experts = router_logits.size(-1)
        ranks = torch.arange(1, num_experts + 1, dtype=torch.float32, device=router_logits.device)
        p = 1.0 / (ranks ** _TokenRouter.zipf.alpha)
        p = p / p.sum()

        batch = router_logits.size(0)
        idx = torch.multinomial(p.expand(batch, -1), top_k, replacement=False)
        w = p.expand(batch, -1).gather(-1, idx)
        w = w / w.sum(dim=-1, keepdim=True)
        return idx, w.to(router_logits.dtype)


class TokenRouter(Enum):
    uniform = _TokenRouter.uniform
    weighted = _TokenRouter.weighted
    multimodal = _TokenRouter.multimodal
    zipf = _TokenRouter.zipf

    def __call__(self, router_logits: Tensor, top_k: int, **kwargs):
        return self.value(router_logits, top_k, **kwargs)
        