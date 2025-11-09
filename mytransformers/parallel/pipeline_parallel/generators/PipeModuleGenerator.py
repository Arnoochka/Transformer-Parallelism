from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator
from mytransformers.parallel.pipeline_parallel.layers import PipeModule
from typing import Dict, Callable
from torch.nn import Module
import torch

class PipeModuleGenerator(ParallelModuleGenerator):
    """
    Обертка для того, чтобы соответствовать базовой структуре генератора 
    
    Args:
        generator (Callable): генератор слоя
        gen_kwargs (Dict): аргументы для генерации
    """
    
    generator: Callable = None
    gen_kwargs: Dict = {}
    def __new__(cls, module: Module, device: torch.device) -> PipeModule:
        pipe_module = cls.generator(module=module, **cls.gen_kwargs)
        return pipe_module.to(device)
