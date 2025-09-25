from torch.nn import Module
from torch import Tensor

class ParallelModule(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: Tensor) -> Tensor:
        return x