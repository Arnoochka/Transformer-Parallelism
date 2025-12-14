import torch
from torch.nn import ModuleList
from torch.distributed import ProcessGroup
from typing import Callable, List, Tuple
from .layers import FakeModule, StrategyModule, PipeBoundaryPointModule
from .Batches import Microbatch, MBatches
from threading import Thread, Lock
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
        self.final_stategy = lambda out: final_strategy(out, *final_strategy_args)
        self.model_forward = model_forward
        self.get_fake_args = fake_args
        
    def set_fake_args(self, mbatch: Microbatch) -> None:
        fake_args_list: List = self.get_fake_args(mbatch.data)
        for module, fake_args in zip(self, fake_args_list):
            if isinstance(module.module, FakeModule):
                module.module.set_gen_args(*fake_args)
           
    @torch.no_grad()
    def forward(self, mbatches: MBatches) -> MBatches:
        TRACKER = get_global_tracker()
        self.k = 0
        def _forward(mbatch: Microbatch) -> None:
            self.set_fake_args(mbatch)
            mbatches[mbatch.idx] = mbatch.compute(self.model_forward) 
        threads: List[Thread] = []
        for mbatch in mbatches:
            TRACKER.snapshot("START mbatch")
            threads.append(Thread(target=_forward, args=(mbatch,), daemon=True))
            threads[-1].start()
            TRACKER.snapshot("END mbatch")  
        for thread in threads:
            thread.join()
        TRACKER.snapshot("END join")
            
        return mbatches

