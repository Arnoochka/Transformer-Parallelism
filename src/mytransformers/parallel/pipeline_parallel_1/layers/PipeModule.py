from mytransformers.parallel.ParallelModule import ParallelModule
from enum import Enum

class PipeRole(Enum):
    dummy = 0
    recv = 1
    send = 2
    compute = 3
    recvAndCompute = 3
    computeAndSend = 4
    none = 5

class PipeModule(ParallelModule):   
    """
    Базовый класс для модулей  конвейерного параллелизма
    """
    def __init__(self, role: PipeRole):
        super().__init__()
        if role == PipeRole.none:
            raise AttributeError(f"comm role is none")
        self.role = role
        
    def __repr__(self) -> str:
        repr = super().__repr__()
        name = self.__class__.__name__
        sub_module = ''.join(repr.split(f"{name}(")[1:])
        return f"{name}(role={self.role}{sub_module}"


        
    