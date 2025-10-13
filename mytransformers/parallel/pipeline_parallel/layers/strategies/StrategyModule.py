from torch.distributed import ProcessGroup
from torch.nn import Module
from typing import Any, Tuple
from torch.distributed import Work

class StrategyModule(Module):
    def __init__(self):
        super().__init__()
        self.tag = 0
    def forward(self,
                output: Any,
                is_send: bool,
                send_group: ProcessGroup,
                recv_group: ProcessGroup) -> Tuple[Any, Any]:
        
        return output, None
        
    def wait(self, worker: Work) -> None:
        worker.wait()
            
    


        
                
    