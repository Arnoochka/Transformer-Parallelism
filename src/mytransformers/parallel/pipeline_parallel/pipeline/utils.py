from typing import Callable, Any
from threading import Condition
from torch import cuda

class MBatch:
    """
    обертка для данных конвейерного параллелизма
    
    Args:
        data (Dict): данные для работы
        idx (int): уникальный индекс микробатча для идентификации
        stream (Stream): поток, в котором будет обрабатываться микробатч
        event (Event): event, который используется для отслеживания завершения обработки микробатча 
    """
    
    def __init__(self,
                 data: Any,
                 idx: int,
                 stream: cuda.Stream,
                 event: cuda.Event):
        self.data = data
        self.idx = idx
        self.stream = stream
        self.event = event
        
    def __repr__(self) -> str:
        return f"MBatch(value={super().__repr__()}, idx={self.idx})"
    
    def wait(self) -> None:
        self.stream.wait_event(self.event)
        
    def compute(self, compute_func: Callable) -> "MBatch":
        with cuda.stream(self.stream):
            self.data = compute_func(**self.data)
            self.event.record(self.stream)
            
        return self


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