from threading import Condition
from typing import Callable, Any

class BaseScheduler:
    def __init__(self) -> None:
        self.state = None
        self.cond = Condition()
        
    def transfer(self, op: Callable[..., Any], op_info: Any, **op_kwargs) -> Any:
        with self.cond:
            while not self.is_allowed(op_info):
                self.cond.wait()
                
            output = op(**op_kwargs)
            
            self.advance(op_info)
            self.cond.notify_all()
            
        return output
            
    def is_allowed(self, op_info: Any) -> bool:
        raise NotImplementedError
    
    def advance(self, op_info: Any) -> None:
        raise NotImplementedError
    
    def reset(self) -> None:
        raise NotImplementedError
    
    def register_alive(self, is_alive: bool) -> None:
        raise NotImplementedError


class RoundRobinScheduler(BaseScheduler):
    def __init__(self):
        super().__init__()
        self.state = 0
        self.curr_num_threads = 1
        self.shift = 0
    
    def is_allowed(self, op_info):
        return self.state == op_info
    
    def advance(self, op_info: int) -> None:
        self.state = (op_info - self.shift + 1) % (self.curr_num_threads - self.shift) + self.shift

    def reset(self) -> None:
        self.state = 0
        self.curr_num_threads = 1
        self.shift = 0
        self.cond.notify_all()
        
    def register_alive(self, is_alive: bool) -> None:
        with self.cond:
            if is_alive:
                self.curr_num_threads += 1
            else:
                self.shift += 1
                if self.state < self.shift:
                    self.state = self.shift
            self.cond.notify_all()
        
    
        