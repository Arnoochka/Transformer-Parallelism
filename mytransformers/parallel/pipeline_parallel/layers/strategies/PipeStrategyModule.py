from torch.distributed import ProcessGroup
from mytransformers.parallel.pipeline_parallel.layers.PipeModule import PipeModule, PipeRole
from torch.nn import Module
from typing import Any

class PipeStrategyModule(PipeModule):
    def __init__(self,
                 role: PipeRole,
                 module: Module,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup,
                 tensor_dim: int = 3):
        super().__init__(role)
        is_send = (role == PipeRole.computeAndSend)
        is_recv = (role  == PipeRole.recv)
        if not (is_send or is_recv):
            raise AttributeError(f"role: {role}, strategy type is not selected")
        
        self.is_send = is_send
        self.send_group = send_group
        self.recv_group = recv_group
        self.module = module
        self.tensor_dim = tensor_dim
        
    def forward(self, *args, **kwargs) -> Any:
        output = self.module(*args, **kwargs)
        return self.transfer_by_strategy(output)
        
    def transfer_by_strategy(self, output) -> Any:
        return output
    


        
                
    