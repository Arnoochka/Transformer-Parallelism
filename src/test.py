import os
import torch
import torch.distributed as dist
from mytransformers import utils
from mytransformers.benchmark import init_global_tracker
from threading import Lock, Thread

lock = Lock()
K = 0
TRACKER = init_global_tracker()

def work():
    with lock:
        dist.barrier()
        TRACKER.snapshot(f"K={K}")
        K += 1

if __name__ == "__main__":
    TRACKER.start()
    utils.init_distributed_cuda()
    rank = dist.get_rank()
    