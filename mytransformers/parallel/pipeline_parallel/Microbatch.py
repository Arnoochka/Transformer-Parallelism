from typing import Dict, Callable
from torch import cuda

class Microbatch:
    """
    обертка для данных конвейерного параллелизма
    
    Args:
        data (Dict): данные для работы
        idx (int): уникальный индекс микробатча для идентификации
        stream (Stream): поток, в котором будет обрабатываться микробатч
        event (Event): event, который используется для отслеживания завершения обработки микробатча 
    """
    
    def __init__(self,
                 data: Dict,
                 idx: int,
                 stream: cuda.Stream,
                 event: cuda.Event):
        self.data = data
        self.idx = idx
        self.stream = stream
        self.event = event
        
    def __repr__(self) -> str:
        return f"Microbatch(value={super().__repr__()}, idx={self.idx})"
    
    def wait(self) -> None:
        self.stream.wait_event(self.event)
        
    def compute(self, compute_func: Callable) -> "Microbatch":
        with cuda.stream(self.stream):
            self.data = compute_func(**self.data)
            self.event.record(self.stream)
            
        return self