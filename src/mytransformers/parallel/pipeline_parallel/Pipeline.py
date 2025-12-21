import torch
from torch.nn import ModuleList
from torch.distributed import ProcessGroup
from typing import Callable, List, Tuple
from .layers import FakeModule, StrategyModule
from .MBatch import MBatch, MBatches
from threading import Thread, Condition

class CondWorker:
    """
    класс для работы с потоками так, чтобы они выполнялись последовательно
    """
    def __init__(self):
        self.cond = Condition()
        self.curr_idx = 0
        
    def __call__(self,
                 mbatch: MBatch,
                 func: Callable[[MBatch], MBatch]) -> MBatch:
        
        with self.cond:
            while self.curr_idx < mbatch.idx:
                self.cond.wait()
                
            mbatch.wait()
            mbatch = func(mbatch)  
            
            self.curr_idx += 1
            self.cond.notify_all()  
            
        return mbatch
    
    def reset(self) -> None:
        self.curr_idx = 0
        
class Pipeline(ModuleList):
    """
    класс конвейерного параллелизма
    
    Механизм работы (для каждого микробатч):
        TODO
            
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
        self.final_cond = CondWorker()
        
    def set_fake_args(self, mbatch: MBatch) -> None:
        fake_args_list: List = self.get_fake_args(mbatch.data)
        for module, fake_args in zip(self, fake_args_list):
            if isinstance(module.module, FakeModule):
                module.module.set_gen_args(*fake_args)
         
           
    @torch.no_grad()
    def forward(self, mbatches: MBatches) -> MBatches:
        self.compute_cond.reset()
        self.final_cond.reset()
        
        def _forward(mbatch: MBatch) -> None:
            def _compute(mbatch: MBatch) -> MBatch:
                self.set_fake_args(mbatch)
                return mbatch.compute(self.model_forward)
                
            mbatches[mbatch.idx] = self.compute_cond(mbatch, _compute)
            
        threads: List[Thread] = []
        for mbatch in mbatches:
            threads.append(Thread(target=_forward, args=(mbatch,), daemon=True))
            threads[-1].start()
            threads[-1].join()
            
        for thread in threads:
            thread.join()
        
        for idx in range(len(mbatches)):
            mbatches[idx].data = self.final_stategy(mbatches[idx].data)
            
        return mbatches

