from mytransformers.parallel.ParallelModule import ParallelModule
from enum import Enum

class PipeRole(Enum):
    dummy = 0
    send = 1
    recv = 2
    compute = 3
    recvAndCompute = 4
    computeAndSend = 5
    none = 6

class PipeModule(ParallelModule):   
    def __init__(self, role: PipeRole):
        super().__init__()
        if role == PipeRole.none:
            raise AttributeError(f"comm role is CommRole.none")
        self.role = role
        
    # def __repr__(self) -> str:
    #     repr = super().__repr__()
    #     name = self.__class__.__name__
    #     sub_module = ''.join(repr.split(f"{name}(")[1:])
    #     return f"{name}(role={self.role}{sub_module}"


        
    