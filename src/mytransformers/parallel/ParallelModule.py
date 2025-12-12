from torch.nn import Module
from typing import Any
import torch

class ParallelModule(Module):
    """
    Базовый параллельный модуль
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Any:
        return None