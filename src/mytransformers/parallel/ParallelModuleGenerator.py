from mytransformers.parallel import ParallelModule
    
class ParallelModuleGenerator:
    """
    Базовый класс генератора.
    """
    def __new__(cls, *args, **kwargs) -> ParallelModule:
        return None