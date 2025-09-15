import torch
import torch.distributed as dist
import torch.cuda as cuda
from typing import Callable, Dict, Optional
from mytransformers.utils import MemoryUnits
import time

class Tracker:
    def __init__(self,
                 group: dist.ProcessGroup,
                 sync_func: Optional[Callable] = None,
                 unit = MemoryUnits.GB):
        self.group = group
        self.sync_func = sync_func
        self.started = False
        self.unit = unit
        self.stats = {}
        
    def start(self) -> None:
        assert not self.started, "tracker has already been started."
        if self.sync_func is not None:
            self.sync_func()
        self.stats = {'name': [], 'time': [], 'memory': [], 'max_memory': []}
        self.started = True
        self.snapshot('start')
        
    def snapshot(self, name: Optional[str] = None) -> None:
        assert self.started, "tracker is not started."
        if self.sync_func is not None:
            self.sync_func()
        if name is None:
            name = f"snap:{len(self.stats['time'])}"
        self.stats['name'].append(name)
        self.stats['time'].append(time.perf_counter())
        self.stats['memory'].append(self._get_memories(cuda.memory_allocated))
        self.stats['max_memory'].append(self._get_memories(cuda.max_memory_allocated))
        cuda.reset_max_memory_allocated(cuda.current_device())
        
    def stop(self) -> Dict:
        assert self.started, "tracker is not started."
        self.started = False
        return self.stats
         
    def _get_memories(self, memory_func: Callable) -> torch.Tensor:
        memory = torch.tensor(memory_func(cuda.current_device()), dtype=torch.float)
        world_size = dist.get_world_size(self.group)
        memories = [torch.zeros_like(memory) for _ in range(world_size)]
        dist.all_gather(memories, memory, self.group)
        return torch.stack(memories) / self.unit