import torch
from torch.nn import ModuleList
from torch.distributed import ProcessGroup
from typing import Callable, List
from mytransformers.parallel.moe_parallel.moe_pipeline.layers import FakeModule, FinalStrategyModule, PipeInnerBoundaryPointModule, PipeComputeModule
from mytransformers.parallel.moe_parallel.moe_pipeline.layers.moe_layers.MoESparseBlockModules import MoeSparseBlockModule
from mytransformers.parallel.moe_parallel.moe_pipeline.pipeline.utils import MBatch, CondWorker
from threading import Thread
from .Scheduler import BaseScheduler
from mytransformers.utils import Logger
        
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
        final_strategy (FinalStrategyModule): финальная передача между процессами
        final_comm_group (ProcessGroup): финальная группа передачи данных
        fake_args (Callable): функция для вычисления арзументов для "фейковых" модулей по микробатчу
        scheduler (BaseScheduler): расписание для коллективных операций
    """
    
    def __init__(self,
                 model_forward: Callable,
                 modules: ModuleList,
                 final_strategy: FinalStrategyModule,
                 final_comm_group: ProcessGroup,
                 fake_args: Callable,
                 scheduler: BaseScheduler):
        super().__init__(modules)
        self.final_stategy = lambda output: final_strategy(output, final_comm_group)
        self.model_forward = model_forward
        self.get_fake_args = fake_args
        
        self.compute_cond = CondWorker()
        self.scheduler = scheduler
        
    def set_fake_args(self, mbatch: MBatch) -> None:
        fake_args_list: List = self.get_fake_args(mbatch.data)
        for module, fake_args in zip(self, fake_args_list):
            if isinstance(module.module, FakeModule):
                module.module.set_gen_args(*fake_args)
                
    def reset(self) -> None:
        self.compute_cond.reset()
        self.scheduler.reset()
        
        for module in self:
            if isinstance(module, PipeInnerBoundaryPointModule) or isinstance(module, PipeComputeModule):
                if isinstance(module, PipeInnerBoundaryPointModule): module.reset()
                for submodule in module.module.modules():
                    if isinstance(submodule, MoeSparseBlockModule):
                        submodule.reset()
         
           
    @torch.no_grad()
    def forward(self, mbatches: List[MBatch], **forward_kwargs) -> List[MBatch]:
        self.reset()
        def _forward(mbatch: MBatch) -> None:
            def _compute(mbatch: MBatch) -> MBatch:
                self.set_fake_args(mbatch)
                self.scheduler.register_alive(True)
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

