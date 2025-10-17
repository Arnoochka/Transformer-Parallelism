from torch.distributed import ProcessGroup
from torch.nn import Module
from typing import Any, List, Optional
from torch.distributed import Work
from itertools import count

GLOBAL_COUNTER = count()

class StrategyModule(Module):
    def __init__(self):
        super().__init__()
        self.tag = None
        self.workers: List[Optional[Work]] = []
    def forward(self,
                output: Any,
                is_send: bool,
                send_group: ProcessGroup,
                recv_group: ProcessGroup) -> Any:
        self.tag = next(GLOBAL_COUNTER)
        self.workers.append(None)
        return output
        
    def wait(self) -> None:
        for worker in self.workers:
            if worker is not None:
                worker.wait()
            
    


        
                
    