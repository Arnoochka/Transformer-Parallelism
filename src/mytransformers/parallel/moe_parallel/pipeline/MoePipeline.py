import torch
from torch.nn import ModuleList
from torch.distributed import ProcessGroup
from typing import Callable, List
from mytransformers.parallel.pipeline_parallel.layers import FakeModule, FinalStrategyModule
from mytransformers.parallel.moe_parallel.layers.sparse_block import MoeSparseBlockModule, MoeSparseBlocFakekDPModule
from mytransformers.parallel.pipeline_parallel.pipeline.utils import MBatch
from mytransformers.parallel.pipeline_parallel.pipeline.Pipeline import Pipeline
from .MoePipeInnerBoundaryPointModule import MoePipeInnerBoundaryPointModule
from .MoeCondWorker import MoeCondWorker
from threading import Thread
from .Scheduler import BaseScheduler
        
        
class MoePipeline(Pipeline):
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
        final_strategy (FinalStrategyModule): финальная передача между процессами
        final_comm_group (ProcessGroup): финальная группа передачи данных
        fake_args (Callable): функция для вычисления арзументов для "фейковых" модулей по микробатчу
        scheduler (BaseScheduler): расписание для коллективных операций
    """
    def __init__(self,
                 model_forward: Callable,
                 modules: ModuleList,
                 reset_modules: List[MoePipeInnerBoundaryPointModule | MoeSparseBlockModule],
                 final_strategy: FinalStrategyModule,
                 final_comm_group: ProcessGroup,
                 fake_args: Callable,
                 scheduler: BaseScheduler):
        super().__init__(model_forward, modules, final_strategy, final_comm_group, fake_args)
        self.reset_modules = reset_modules
        self.scheduler = scheduler
        self.compute_cond = MoeCondWorker(scheduler)
        for module in self:
            if isinstance(module, MoePipeInnerBoundaryPointModule):
                module.compute_cond = self.compute_cond
        
    def set_fake_args(self, mbatch: MBatch) -> None:
        fake_args_list: List = self.get_fake_args(mbatch.data)
        for module, fake_args in zip(self, fake_args_list):
            if isinstance(module.module, FakeModule):
                module.module.set_gen_args(*fake_args)
            elif isinstance(module.module, MoeSparseBlocFakekDPModule):
                module.module.fake_module.set_gen_args(*fake_args)
                
    def start(self, num_mbatches: int) -> None:
        self.scheduler.reset()
        for module in self.reset_modules:
            module.reset()
        self.compute_cond.start(num_mbatches)
              
    @torch.no_grad()
    def forward(self, mbatches: List[MBatch], **forward_kwargs) -> List[MBatch]:
        self.start(len(mbatches))
        def _forward(mbatch: MBatch) -> None:
            def _compute(mbatch: MBatch) -> MBatch:
                self.set_fake_args(mbatch)
                mbatch.data.update(forward_kwargs)
                return mbatch.compute(self.model_forward)  
            
            mbatches[mbatch.idx] = self.compute_cond(mbatch, _compute)
            
        threads: List[Thread] = []
        for mbatch in mbatches:
            threads.append(Thread(target=_forward, args=(mbatch,), daemon=True))
            threads[-1].start()
            
        for thread in threads:
            thread.join()
            self.scheduler.register_alive(False)
            
        for idx in range(len(mbatches)):
            mbatches[idx].data = self.final_stategy(mbatches[idx].data)
            
        return mbatches

