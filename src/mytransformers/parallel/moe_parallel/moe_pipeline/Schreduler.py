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


class RoundRobinScheduler(BaseScheduler):
    def __init__(self) -> None:
        super().__init__(init_state=0)
        self.curr_num_threads = 0
    
    def is_allowed(self, op_info: Tuple[int, bool]) -> bool:
        stage_idx, is_point = op_info
        return self.state == stage_idx
    
    def advance(self, op_info: Tuple[int, bool]) -> None:
        stage_idx, is_point = op_info
        self.state = (self.state + 1)
        