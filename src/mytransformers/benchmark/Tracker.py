import torch
import torch.distributed as dist
import torch.cuda as cuda
from typing import Callable, Optional
from mytransformers import utils
import pandas as pd
import time

class Tracker:
    def __init__(self,
                 group: Optional[dist.ProcessGroup] = None,
                 sync_func: Optional[Callable] = None,
                 unit = utils.MemoryUnits.GB):
        self.group = group
        self.sync_func = sync_func
        self.is_started = False
        self.unit = unit
        self.stats = None
        
    def start(self) -> None:
        assert not self.is_started, "tracker has already been started."
        if self.sync_func is not None:
            self.sync_func()
        self.stats = {'name': [], 'time': [], 'memory': [], 'max_memory': []}
        self.is_started = True
        self.snapshot('start')
        
    def snapshot(self, name: Optional[str] = None) -> None:
        assert self.is_started, "tracker is not started."
        if self.sync_func is not None:
            self.sync_func()
        if name is None:
            name = f"snap:{len(self.stats['time'])}"
        self.stats['name'].append(name)
        self.stats['time'].append(time.perf_counter())
        self.stats['memory'].append(self._get_memories(cuda.memory_allocated))
        self.stats['max_memory'].append(self._get_memories(cuda.max_memory_allocated))
        cuda.reset_max_memory_allocated(cuda.current_device())
        utils.Logger.log_all_device(f"STATS: name={self.stats['name'][-1]}")
        
    def stop(self) -> pd.DataFrame:
        assert self.is_started, "tracker is not started."
        self.snapshot("stop")
        self.is_started = False
        world_size = dist.get_world_size(self.group)
        flat_stats = {'name': self.stats['name'],
                      'time': [t - self.stats['time'][0] for t in self.stats['time']]}
        
        for k in range(len(self.stats['time'])):
            flat_stats['time'][k] = self.stats['time'][k] - self.stats['time'][0]
        
        for i in range(world_size):
            flat_stats[f'memory_gpu_{i}'] = [mem[i].item() for mem in self.stats['memory']]
            flat_stats[f'max_memory_gpu_{i}'] = [mem[i].item() for mem in self.stats['max_memory']]
            
        df = pd.DataFrame(flat_stats)
        df = df.set_index("name")
        return df
         
    def _get_memories(self, memory_func: Callable) -> torch.Tensor:
        memory = memory_func(cuda.current_device())
        memory_tensor = torch.tensor(memory, dtype=torch.float, device='cuda')
        world_size = dist.get_world_size(self.group)
        memories = [torch.zeros_like(memory_tensor) for _ in range(world_size)]
        dist.all_gather(memories, memory_tensor, self.group)
        return torch.stack(memories) / self.unit.value
    
TRACKER: Optional[Tracker] = None
def get_global_tracker() -> Tracker:
    global TRACKER
    return TRACKER

def init_global_tracker(group: Optional[dist.ProcessGroup] = None,
                        sync_func: Optional[Callable] = None,
                        unit = utils.MemoryUnits.GB) -> Tracker:
    global TRACKER
    TRACKER = Tracker(group, sync_func, unit)
    return TRACKER
    
    