from typing import Callable, List, Any
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
    
class MBatches:
    def __init__(self, split_func: Callable[..., List[MBatch]]):
        self.mbatches: List[MBatch] = None
        self.split = split_func
    
    def __call__(self, *args, **kwargs) -> "MBatches":
        self.mbatches = self.split(*args, **kwargs)
        return self
    
    def __iter__(self):
        for mbatch in self.mbatches:
            mbatch.wait()
            yield mbatch

    def __getitem__(self, i: int) -> MBatch:
        mbatch = self.mbatches[i]
        mbatch.wait()
        return mbatch

    def __setitem__(self, i: int, value: MBatch) -> None:
        self.mbatches[i] = value
        
    
        

        
    