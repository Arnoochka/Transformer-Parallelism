from typing import Any
from mytransformers.parallel.ParallelModuleGenerator import ParallelModuleGenerator

class PipeGenerator(ParallelModuleGenerator):
    def __new__(cls, *args, **kwargs) -> Any:
        return None