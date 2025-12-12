import torch
from torch.nn import ModuleList
from torch.distributed import ProcessGroup
from typing import Callable, List, Any, Tuple
from .layers import FakeModule, StrategyModule
from .Batches import Microbatch
from threading import BoundedSemaphore
from mytransformers.benchmark import get_global_tracker
        
class Pipeline(ModuleList):
    """
    класс конвейерного параллелизма
    
    Механизм работы (для каждого микробатч):
        1. Дожидаемся, что текущий микробатч прошел все стадии при предыдущем вызове forward
        2. Дожидаемся, когда предыдущий микробатч пройдет стадию
        3. Устанавливаем арзументы для "фейковых" тензоров
        4. Запускаем микробатч
            
    Args:
        model_forward (Callable): forward ункция исходной модели
        modules (Iterable[Module]): все подмененные слои модели
        fake_args (Callable): функция для вычисления арзументов для "фейковых" модулей по микробатчу
    """
    
    def __init__(self,
                 model_forward: Callable,
                 modules: ModuleList,
                 final_strategy: StrategyModule,
                 final_strategy_args: Tuple[bool, ProcessGroup, ProcessGroup],
                 fake_args: Callable):
        super().__init__(modules)
        self.final_stategy = self.build_final_strategy(final_strategy, final_strategy_args)
        self.compute = self.build_compute(model_forward)
        self.get_fake_args = fake_args
        self.can_push = BoundedSemaphore()
        
    def set_fake_args(self, mbatch: Microbatch) -> None:
        fake_args_list: List = self.get_fake_args(mbatch.data)
        for module, fake_args in zip(self, fake_args_list):
            if isinstance(module.module, FakeModule):
                module.module.set_gen_args(*fake_args)
                
    def build_final_strategy(self, 
                             final_strategy: StrategyModule,
                             final_strategy_args: Tuple[bool, ProcessGroup, ProcessGroup]
                             ) -> Callable:
        def _final_stategy(output: Any) -> Any:
            return final_strategy(output, *final_strategy_args)
        
        return _final_stategy
                
    def build_compute(self, model_forward: Callable) -> Callable:
        def _compute(*args, **kwargs) -> Any:
            TRACKER = get_global_tracker()
            TRACKER.snapshot("START compute")
            output = model_forward(*args, **kwargs)
            TRACKER.snapshot("END compute")
            self.can_push.release()
            self.final_stategy(output)
            return output
        
        return _compute
            
    @torch.no_grad()
    def forward(self, mbatches: List[Microbatch]) -> List[Microbatch]:
        TRACKER = get_global_tracker()
        TRACKER.snapshot("START forward")
        for idx, mbatch in enumerate(mbatches):
            mbatch.wait()
            self.can_push.acquire()
            self.set_fake_args(mbatch)
            mbatches[idx] = mbatch.compute(self.compute)
        TRACKER.snapshot("END forward")   
        return mbatches

