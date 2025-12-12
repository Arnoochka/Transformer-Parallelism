import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from .TPModuleGenerator import TPModuleGenerator
from mytransformers.parallel.tensor_parallel.layers import TPColumnEmbedding, TPRowEmbedding
from torch.nn import Embedding


class TPColumnEmbeddingGenerator(TPModuleGenerator):
    """
    генерирует TPColumnEmbedding из torch.nn.Embedding

    Args:
        use_all_gather (bool, optional): Если ``True``, объединяет результаты со всех устройств
            с помощью операции all_gather. По умолчанию ``True``.
        tp_group (ProcessGroup): Группа процессов для тензорного параллелизма.
    """
    use_all_gather: bool = True
    tp_group: ProcessGroup = None

    @torch.no_grad()
    def __new__(cls, module: Embedding, device: torch.device) -> TPColumnEmbedding:
        if TPColumnEmbeddingGenerator.already_converted(module):
            return module

        tp_size = dist.get_world_size(cls.tp_group)
        rank = dist.get_rank(cls.tp_group)

        num_embeddings = module.num_embeddings
        embedding_dim = module.embedding_dim
        assert embedding_dim % tp_size == 0, \
            "embedding_dim must be divisible by tp_size"

        layer = TPColumnEmbedding(
            num_embeddings,
            embedding_dim,
            cls.tp_group,
            module.padding_idx,
            module.max_norm,
            module.norm_type,
            module.sparse,
            use_all_gather=cls.use_all_gather
        )

        weight = module.weight.chunk(tp_size, dim=1)[rank]
        layer.weight.copy_(weight.contiguous())
        return layer.to(device)


class TPRowEmbeddingGenerator(TPModuleGenerator):
    """
    генерирует TPRowEmbedding из torch.nn.Embedding

    Args:
        use_all_reduce (bool, optional): Если ``True``, объединяет результаты с помощью
            операции all_reduce. По умолчанию ``True``.
        tp_group (ProcessGroup): Группа процессов для тензорного параллелизма.
    """
    use_all_reduce: bool = True
    tp_group: ProcessGroup

    @torch.no_grad()
    def __new__(cls, module: Embedding, device: torch.device) -> TPRowEmbedding:
        if TPRowEmbeddingGenerator.already_converted(module):
            return module

        tp_size = dist.get_world_size(cls.tp_group)
        rank = dist.get_rank(cls.tp_group)

        num_embeddings = module.num_embeddings
        embedding_dim = module.embedding_dim
        assert num_embeddings % tp_size == 0, \
            "num_embeddings must be divisible by tp_size"

        layer = TPRowEmbedding(
            num_embeddings,
            embedding_dim,
            cls.tp_group,
            module.padding_idx,
            module.max_norm,
            module.norm_type,
            module.sparse,
            use_all_reduce=cls.use_all_reduce
        )

        weight = module.weight.chunk(tp_size, dim=0)[rank]
        layer.weight.copy_(weight.contiguous())

        return layer.to(device)
