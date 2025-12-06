import torch
from torch.nn import ModuleList
from typing import Callable, List
from .layers import FakeModule
from .Microbatch import Microbatch
from threading import Event
        
class Pipeline(ModuleList):
    """
    класс конвейерного параллелизма
    
    Механизм работы (для каждого микробатч):
        1. Устанавливаются арзементы для генерации "фейковых выходах"
        2. дожидаемся, когда от финального слоя придет callback, о том, что предыдущий микробатч работу завершил 
        3. Дожидаемся, что текущий микробатч прошел все стадии при предыдущем вызове forward
        4. запускаем проход текущего микробатча в новом потоке
        
    Args:
        model_forward (Callable): forward ункция исходной модели
        modules (Iterable[Module]): все подмененные слои модели
        fake_args (Callable): функция для вычисления арзументов для "фейковых" модулей по микробатчу
        num_microbatches (int): число микробатчей
    """
    
    def __init__(self,
                 model_forward: Callable,
                 modules: ModuleList,
                 fake_args: Callable,
                 num_microbatches: int):
        super().__init__(modules)
        self.model_forward = model_forward
        self.get_fake_args = fake_args
        self.num_microbatches = num_microbatches
        self.can_push = Event()
        
        self.set_callback()
        
    def set_callback(self) -> None:
        def _make_callback(is_finished: bool):
            if is_finished:
                self.can_push.set()
            else:
                self.can_push.clear()
                           
        self[0].set_callback(_make_callback)
        self[-1].set_callback(_make_callback)
        
    def set_fake_args(self, mbatch: Microbatch) -> None:
        fake_args_list: List = self.get_fake_args(mbatch.data)
        for module, fake_args in zip(self, fake_args_list):
            if isinstance(module.module, FakeModule):
                module.module.set_gen_args(**fake_kwargs)
            
    @torch.no_grad()
    def forward(self, mbatches: List[Microbatch]) -> List[Microbatch]:
        self.can_push.set()
        
        for idx, mbatch in enumerate(mbatches):
            self.set_fake_args(mbatch)
            self.can_push.wait()
            mbatch.wait()
            mbatch[idx] = mbatch.compute(self.model_forward)
                
        return mbatches
    
    
        
    
    
        
        
        