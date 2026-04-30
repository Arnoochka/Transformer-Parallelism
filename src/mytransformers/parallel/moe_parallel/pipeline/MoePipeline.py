import torch
from torch.nn import ModuleList
from torch.distributed import ProcessGroup
from typing import Callable, List
from mytransformers.parallel.pipeline_parallel.layers import FinalStrategyModule
from mytransformers.parallel.moe_parallel.layers.MoeLayers import MoeFakeLayer, MoeComputeLayer
from mytransformers.parallel.pipeline_parallel.layers import FakeModule
from mytransformers.parallel.pipeline_parallel.pipeline.utils import MBatch
from mytransformers.parallel.pipeline_parallel.pipeline.Pipeline import Pipeline
from .MoePipeInnerBoundaryPointModule import MoePipeInnerBoundaryPointModule
from .MoeCondWorker import MoeCondWorker
from threading import Thread
from .Scheduler import BaseScheduler
        
        
class MoePipeline(Pipeline):

    def __init__(self,
                 model_forward: Callable,
                 modules: ModuleList,
                 inner_boundary_points: List[int],
                 final_strategy: FinalStrategyModule,
                 final_comm_group: ProcessGroup,
                 fake_args: Callable,
                 scheduler: BaseScheduler):
        super().__init__(model_forward, modules, final_strategy, final_comm_group, fake_args)
        self.scheduler = scheduler
        self.compute_cond = MoeCondWorker(scheduler)
        for idx, module in enumerate(self):
            if idx in inner_boundary_points:
                module.set_notify(self.compute_cond.notify)
        
    def set_fake_args(self, mbatch: MBatch) -> None:
        fake_args_list: List = self.get_fake_args(mbatch.data)
        for module, fake_args in zip(self, fake_args_list):
            if isinstance(module, MoeFakeLayer):
                module.set_gen_args(*fake_args)
            elif isinstance(module.module, FakeModule):
                module.module.set_gen_args(*fake_args)
                
    def start(self, num_mbatches: int) -> None:
        self.scheduler.reset()
        for module in self:
            if isinstance(module, MoeComputeLayer):
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

