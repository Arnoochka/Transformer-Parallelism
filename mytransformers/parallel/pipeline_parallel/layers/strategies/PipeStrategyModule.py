from torch.distributed import ProcessGroup
from mytransformers.parallel.pipeline_parallel.layers.PipeModule import PipeModule, PipeRole
from torch.nn import Module

class PipeStrategyModule(PipeModule):
    def __init__(self,
                 role: PipeRole,
                 module: Module,
                 send_group: ProcessGroup,
                 recv_group: ProcessGroup,
                 tensor_dim: int = 3):
        super().__init__(role)
        is_send = role in (PipeRole.send, PipeRole.computeAndSend)
        is_recv = role in (PipeRole.recv, PipeRole.recvAndCompute)
        if not (is_send or is_recv):
            raise AttributeError(f"role: {role}, but it is send strategy")
        
        self.is_send = is_send
        self.send_group = send_group
        self.recv_group = recv_group
        self.module = module
        self.tensor_dim = tensor_dim
        
                
    