import torch
from .fake_modules import FakeModule
from .PipeModule import PipeModule, PipeRole
from typing import Any

class PipeFakeModule(PipeModule):
    """
    Фейковвый модуль для конвейерного параллелизма.
    
    Задача: подмена слоя на GPU, где на данном этапе вычисления не происходят

    Args:
        fame_module (FakeModule): тип фейкового модуля, который для подмены
    """
    def __init__(self, fake_module: FakeModule):
        super().__init__(PipeRole.dummy)
        self.fake_module = fake_module
        
    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Any:
        return self.fake_module()