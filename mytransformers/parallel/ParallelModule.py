from torch.nn import Module
from typing import Any

class ParallelModule(Module):
    """
    Базовый параллельный модуль
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, *args, **kwargs) -> Any:
        return None