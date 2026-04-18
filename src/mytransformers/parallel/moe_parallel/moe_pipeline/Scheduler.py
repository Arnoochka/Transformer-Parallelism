from threading import Condition
from typing import Callable, Any, Tuple
        


class BaseScheduler:
    def __init__(self, init_state: Any) -> None:
        self.state = init_state
        self.cond = Condition()
        
    def transfer(self, op: Callable[..., Any], op_info: Any, *op_args) -> Any:
        with self.cond:
            while not self.is_allowed(op_info):
                self.cond.wait()
                
        output = op(*op_args)
            
        with self.cond:
            self.advance(op_info)
            self.cond.notify_all()
            
        return output
            
    def is_allowed(self, op_info: Any) -> bool:
        raise NotImplementedError
    
    def advance(self, op_info: Any) -> None:
        raise NotImplementedError
    
    def reset(self) -> None:
        raise NotImplementedError


class InternalScheduler(BaseScheduler):
    def __init__(self) -> None:
        super().__init__(init_state=0)
        self.alive = [False, False]
    
    def is_allowed(self, op_info: int) -> bool:
        return self.state == op_info
    
    def advance(self, op_info: int) -> None:
        if self.alive[(self.state + 1) % 2]:
            self.state = (self.state + 1) % 2
        
    def register_alive(self, stage_idx: int, is_alive: bool) -> None:
        self.alive[stage_idx] = is_alive
        
    
        