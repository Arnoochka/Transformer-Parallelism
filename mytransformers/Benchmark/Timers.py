import time
from typing import Callable, Optional, List
from typing import Dict


class Timer:
    """An internal timer."""

    def __init__(self, name: str):
        self.name = name
        self.started = False
        self.start_time = None
        self.costs = []

    def start(self, sync_func: Optional[Callable] = None):
        """Start the timer."""
        assert not self.started, f"timer {self.name} has already been started."
        if sync_func:
            sync_func()

        self.start_time = time.perf_counter()
        self.started = True

    def stop(self, sync_func: Callable = None):
        """Stop the timer."""
        assert self.started, f"timer {self.name} is not started."
        if sync_func:
            sync_func()

        stop_time = time.perf_counter()
        self.costs.append(stop_time - self.start_time)
        self.started = False

    def reset(self):
        """Reset timer."""
        self.started = False
        self.start_time = None
        self.costs = []

class Timers:
    """A group of timers."""

    def __init__(self, names: Optional[List[str]] = None):
        self.timers = {}
        if names is not None:
            for name in names:
                self.timers[name] = Timer(name)
                
    def __call__(self, name: str):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def __contains__(self, name: str):
        return name in self.timers
    
    def start_timer(self, name: str) -> None:
        assert name in self.timers, "this timer is not exist"
        self.timers[name].start()
        
    def stop_timer(self,
                   name: str,
                   sync_func: Optional[Callable] = None):
        assert name in self.timers, "this timer is not exist"
        self.timers[name].stop(sync_func)
        
    def reset_timers(self) -> None:
        for timer in self.timers.values():
            timer.reset()