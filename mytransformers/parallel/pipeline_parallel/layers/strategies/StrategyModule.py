from torch.distributed import ProcessGroup
from torch.nn import Module
from typing import Any
from itertools import count

class StrategyModule(Module):
    """
    Базовый класс стратегий
    """
    def __init__(self):
        super().__init__()
        
    def forward(self,
                output: Any,
                is_send: bool,
                send_group: ProcessGroup,
                recv_group: ProcessGroup) -> Any:
        
        """
        Args:
            output (Any): выход, который необходимо передать не следующий процесс
            is_send (bool): является ли процесс отправителем
            send_group (ProcessGroup): группа, процессов, которая отправляет данные
            recv_group (ProcessGroup): группа процессов, которая принимает данные
            
        Returns:
            Any: выход, который необходимо быдло передать (output)
        """
        return output
    


        
                
    