import torch
from torch.nn import ModuleList
from torch.distributed import ProcessGroup
from typing import Callable, List, Tuple
from mytransformers.parallel.pipeline_parallel.layers import FakeModule, StrategyModule
from .utils import MBatch, CondWorker
from threading import Thread
        
class Pipeline(ModuleList):
    """
    класс конвейерного параллелизма
    
    Механизм работы:
        1. запуск потока с mbatch. Это необходимо для пернкрытия CPU операций
        2. внутри каждого потока выполнение оборачивается в cuda поток. Это необходимо для перекрытия CPU-GPU операций
        3. Для того, чтобы потоки не мешали друг другу, вычисления в потоках происходят последовательно при помощи CondWorker
        4. Применям финальную стратегию, чтобы на необходимых GPU были актуальные данные
            
    Args:
        model_forward (Callable): forward ункция исходной модели
        modules (ModuleList): все подмененные слои модели
        final_strategy (StrategyModule): финальная передача между процессами
        final_strategy_args (Tuple[bool, ProcessGroup, ProcessGroup]): аргументы финальной передачи
        fake_args (Callable): функция для вычисления арзументов для "фейковых" модулей по микробатчу
    """
    
    def __init__(self,
                 model_forward: Callable,
                 modules: ModuleList,
                 final_strategy: StrategyModule,
                 final_strategy_args: Tuple[bool, ProcessGroup, ProcessGroup],
                 fake_args: Callable):
        super().__init__(modules)
        self.final_stategy = lambda output: final_strategy(output, *final_strategy_args)
        self.model_forward = model_forward
        self.get_fake_args = fake_args
        
        self.compute_cond = CondWorker()
        
    def set_fake_args(self, mbatch: MBatch) -> None:
        fake_args_list: List = self.get_fake_args(mbatch.data)
        for module, fake_args in zip(self, fake_args_list):
            if isinstance(module.module, FakeModule):
                module.module.set_gen_args(*fake_args)
         
           
    @torch.no_grad()
    def forward(self, mbatches: List[MBatch]) -> List[MBatch]:
        
        self.compute_cond.reset()
        def _forward(mbatch: MBatch) -> None:
            def _compute(mbatch: MBatch) -> MBatch:
                self.set_fake_args(mbatch)
                return mbatch.compute(self.model_forward)  
            mbatches[mbatch.idx] = self.compute_cond(mbatch, _compute)
            
        threads: List[Thread] = []
        for mbatch in mbatches:
            threads.append(Thread(target=_forward, args=(mbatch,), daemon=True))
            threads[-1].start()
            
        for thread in threads:
            thread.join()
        
        for idx in range(len(mbatches)):
            mbatches[idx].data = self.final_stategy(mbatches[idx].data)
            
        return mbatches

