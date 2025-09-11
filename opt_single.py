import torch
import torch.distributed as dist
from transformers import pipeline
from mytransformers import tp_custom
from mytransformers import utils

if __name__ == "__main__":
    pipe = pipeline("text-generation",
                    model="facebook/opt-350m",
                    device=torch.device('cuda'),
                    torch_dtype=torch.float32)
    output = pipe('Input String', use_cache=False)
    print(f"peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f}")
    print(f"output: {output}")