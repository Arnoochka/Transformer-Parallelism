import os
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import nn
from torch.distributed.rpc import RRef
from typing import List

# ---------------- RPC -----------------
def init_rpc(rank, world_size):
    worker_name = f"worker{rank}"
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=300
    )
    rpc.init_rpc(
        name=worker_name,
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc_backend_options
    )
    return worker_name

# ---------------- Remote Layer -----------------
class RemoteLayer(nn.Module):
    def __init__(self, layer: nn.Module, worker_name: str):
        super().__init__()
        self.layer_rref = RRef(layer)
        self.worker_name = worker_name

    def forward_async(self, x):
        # Асинхронный вызов слоя
        return rpc.rpc_async(self.worker_name, _forward, args=(self.layer_rref, x))

def _forward(layer_rref: RRef, x: torch.Tensor):
    layer = layer_rref.local_value()
    return layer(x)

# ---------------- Pipeline Model -----------------
class PipelineModel(nn.Module):
    def __init__(self, layers: List[RemoteLayer]):
        super().__init__()
        self.layers = layers

    def forward(self, x_list: List[torch.Tensor]):
        """
        x_list: список батчей для pipeline
        """
        num_batches = len(x_list)
        # Каждому батчу создаём future
        futures = [torch.futures.Future() for _ in range(num_batches)]
        
        # Инициализируем pipeline
        for i, x in enumerate(x_list):
            fut = self.layers[0].forward_async(x)
            futures[i] = fut

        # Проходим через остальные слои
        for layer in self.layers[1:]:
            for i in range(num_batches):
                # Ждём результат предыдущего слоя
                futures[i] = layer.forward_async(futures[i].wait())
        
        # Ждём финальных результатов
        results = [fut.wait() for fut in futures]
        return results

# ---------------- Main -----------------
def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    worker_name = init_rpc(rank, world_size)
    all_workers = [f"worker{i}" for i in range(world_size)]

    if rank == 0:
        # Создаём модель и распределяем слои по workers
        model = nn.Sequential(
            nn.Linear(10, 20).cuda(0),
            nn.ReLU().cuda(1),
            nn.Linear(20, 5).cuda(1)
        )

        # Каждый слой на своём worker
        remote_layers = [RemoteLayer(layer, worker_name) for layer, worker_name in zip(model.children(), all_workers)]
        pipeline_model = PipelineModel(remote_layers)

        # Пример батчей
        x_list = [torch.randn(4, 10).cuda(0) for _ in range(4)]
        outputs = pipeline_model(x_list)
        for i, y in enumerate(outputs):
            print(f"Batch {i} output:", y)

    rpc.shutdown()

if __name__ == "__main__":
    main()