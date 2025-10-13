from torch.nn import Module
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Tuple, Dict
from mytransformers.parallel.pipeline_parallel.layers import (PipeFakeModule, PipeModule,
                                                              PipeRole,
                                                              PipeStrategyModule,
                                                              StrategyModule,
                                                              FakeModule)
        
    

class StrategyModuleGenerator:
    def __new__(cls,
                role: PipeRole,
                module: Module,
                group_info: Tuple[ProcessGroup, List[int]],
                next_role: PipeRole,
                next_module: Module,
                next_group_info: Tuple[ProcessGroup, List[int]],
                fake_module: FakeModule,
                strategy: StrategyModule,
                strategy_kwargs: Dict = {}) -> PipeModule:
        rank = dist.get_rank()
        group, ranks = group_info
        next_group, next_ranks = next_group_info
        if rank in ranks:
            return PipeStrategyModule(role,
                                      module,
                                      group,
                                      next_group,
                                      strategy(**strategy_kwargs))
        elif rank in next_ranks:
            return PipeStrategyModule(next_role,
                                      next_module,
                                      group,
                                      next_group,
                                      strategy(**strategy_kwargs))
        else:
            return PipeFakeModule(fake_module)