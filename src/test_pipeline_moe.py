import os
import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
from typing import Tuple
from mytransformers import utils
from mytransformers.parallel import moe
from mytransformers.parallel.moe_parallel import moe_pp
from mytransformers.benchmark import moe_test


if __name__ == "__main__":
    model = moe_test.TestModel(moe_test.Config)
    print(model)

    # utils.init_distributed_cuda()
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()

    # device = torch.cuda.current_device()


    # stages = [
    #     (utils.create_group([0]), [0]),
    #     (utils.create_group([1]), [1]),
    # ]

    # inner_comm_groups = [
    #     utils.create_group([0, 1]),
    #     ]
    
    