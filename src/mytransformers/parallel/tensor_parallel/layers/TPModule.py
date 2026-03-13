from torch.distributed import ProcessGroup
from mytransformers.parallel.ParallelModule import ParallelModule
from torch import Tensor

class TPModule(ParallelModule):
    """
    базовый класс для тензорных параллельных модулей
    """
    def __init__(self, tp_group: ProcessGroup):
        super().__init__()
        self.tp_group = tp_group
        
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)