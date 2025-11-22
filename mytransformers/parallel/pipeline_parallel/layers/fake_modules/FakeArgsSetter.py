from typing import Any, Callable, List
from .FakeModule import FakeModule


class FakeArgsSetter:
    def __init__(self, get_args_funcs: List[Callable]) -> None:
        self.get_args_funcs = get_args_funcs
    
    def __call__(self, output: Any) -> Any:
        pass
    
    