from typing import Callable
from mytransformers.parallel.pipeline_parallel.pipeline.utils import MBatch, CondWorker
from .Scheduler import BaseScheduler

class MoeCondWorker(CondWorker):
    def __init__(self,
                 scheduler: BaseScheduler):
        self.scheduler = scheduler
        self.num_mbatches: int = None
        super().__init__()
    def __call__(self,
                 mbatch: MBatch,
                 func: Callable[[MBatch], MBatch]) -> MBatch:
        
        with self.cond:
            while self.curr_idx < mbatch.idx:
                self.cond.wait()
                
        mbatch.wait()
        mbatch = func(mbatch)  
            
        return mbatch

    def notify(self) -> None:
        print("NOTIFY")
        if self.num_mbatches - 1 > self.curr_idx:
            with self.cond:
                self.scheduler.register_alive(True)
                self.curr_idx += 1
                self.cond.notify_all()  
            
    def start(self, num_mbatches: int) -> None:
        self.num_mbatches = num_mbatches
        self.reset()
        self.scheduler.register_alive(True)