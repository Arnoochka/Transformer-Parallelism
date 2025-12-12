import os
from typing import List, Any
import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
from enum import Enum
from torch.distributed import ProcessGroup


# Глобальные переменные, используемые для распределённой инициализации
GROUP: ProcessGroup = None
RANKS: List[int] = None
BACKEND: str = None


class MemoryUnits(Enum):
    """
    Единицы измерения памяти для расчёта размера модели.

    Attributes:
        GB (int): Гигабайт (1024 ** 3).
        MB (int): Мегабайт (1024 ** 2).
        KB (int): Килобайт (1024).
    """
    GB = 1024**3
    MB = 1024**2
    KB = 1024


class Logger:
    """
    Класс для логирования данных в распределённых системах.

    Предоставляет методы для вывода информации с одного или всех ранков.
    """

    @staticmethod
    def log_main_device(log: Any) -> None:
        """
        Выводит данные только с процесса с определённым рангом.

        Args:
            log (Any): Данные, которые необходимо вывести.
            rank (Optional[int], optional): 
                Ранк процесса, с которого нужно вывести данные. 
                Если не указан, по умолчанию используется текущий процесс (``dist.get_rank()``).
        """
        rank = dist.get_rank()
        if rank == 0:
            if not isinstance(log, str):
                log = f"{log}"
            print(log)

    @staticmethod
    def log_all_device(log: Any) -> None:
        """
        Выводит данные со всех рангов, добавляя к выводу номер текущего ранга.

        Args:
            log (Any): Данные, которые необходимо вывести.
        """
        print(f"---device:{dist.get_rank()}---:\n{log}")
        
    @staticmethod
    def log(log: Any, rank: int) -> None:
        """
        Выводит данные c конкретного ранга, добавляя к выводу номер текущего ранга.

        Args:
            log (Any): Данные, которые необходимо вывести.
            rank (int): ранг, с которого необходимо вывести данные.
        """
        if dist.get_rank() == rank:
            print(f"---device:{rank}---:\n{log}")
        


def get_prompts(filename: str) -> List[str]:
    """
    Загружает список строк из текстового файла.

    Args:
        filename (str): Путь к файлу, содержащему промпты (один на строку).

    Returns:
        List[str]: Список строк (промптов).
    """
    with open(filename, 'r') as file:
        promts = [line for line in file]
        return promts


def torch_round(tensor: Tensor, decimals: int) -> Tensor:
    """
    Округляет значения тензора до указанного числа знаков после запятой.

    Args:
        tensor (Tensor): Исходный тензор.
        decimals (int): Количество знаков после запятой.

    Returns:
        Tensor: Округлённый тензор.
    """
    multiplier = 10 ** decimals
    return torch.round(tensor * multiplier) / multiplier


def get_model_size(model: Module, unit: MemoryUnits = MemoryUnits.GB) -> float:
    """
    Вычисляет общий размер модели (параметров и буферов) в указанных единицах.

    Args:
        model (Module): PyTorch-модель.
        unit (MemoryUnits, optional): Единицы измерения памяти (по умолчанию ``MemoryUnits.GB``).

    Returns:
        float: Размер модели в выбранных единицах измерения.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size
    return total_size / unit.value


def init_distributed(backend: str = 'nccl') -> None:
    """
    Инициализирует распределённое окружение PyTorch.

    Устанавливает глобальные переменные ``GROUP``, ``RANKS`` и ``BACKEND``.

    Args:
        backend (str, optional): Бэкенд для распределённых вычислений (по умолчанию ``'nccl'``).
    """
    global GROUP, BACKEND, RANKS
    BACKEND = backend
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=BACKEND, world_size=world_size, rank=rank)
    RANKS = [k for k in range(world_size)]
    GROUP = dist.new_group(ranks=RANKS, backend=BACKEND)
    torch.manual_seed(0)


def init_distributed_cuda() -> None:
    """
    Инициализирует распределённое окружение с использованием CUDA.

    Устанавливает устройство CUDA в соответствии с текущим рангом и синхронизирует сид.
    """
    init_distributed('nccl')
    rank = int(os.environ["RANK"])
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(rank)


def create_group(ranks: List[int]) -> ProcessGroup:
    """
    Создаёт новую группу процессов для распределённого взаимодействия.

    Args:
        ranks (List[int]): Список ранков, входящих в группу.

    Returns:
        ProcessGroup: Новый объект группы процессов.
    """
    group = dist.new_group(ranks=ranks, backend=BACKEND)
    return group
