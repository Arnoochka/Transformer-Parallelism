from mytransformers.parallel import ParallelModule
from torch.nn import Module
import torch
    
class ParallelModuleGenerator(Module):
    """
    Базовый класс генератора.
    
    входные значения для генератора:
        module: модуль, которые будет заменяться
        device: устройство, на котором сгенерированный модкль должен быть
    """
    @torch.no_grad()
    def __new__(cls, module: Module, device: torch.device) -> ParallelModule:
        return ParallelModule()