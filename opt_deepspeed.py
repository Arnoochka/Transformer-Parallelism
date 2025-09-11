import torch
import deepspeed.comm as comm
import deepspeed
from transformers import pipeline
from mytransformers import utils

if __name__ == "__main__":
    pipe = pipeline("text-generation",
                    model="facebook/opt-350m",
                    device=torch.device('cuda'),
                    torch_dtype=torch.float32)
    comm.init_distributed('nccl')
    world_size = comm.get_world_size()
    local_rank = comm.get_rank()
    pipe.model = deepspeed.init_inference(pipe.model,
                                          tensor_parallel = {'tp_size': world_size},
                                          dtype=torch.float32,
                                          replace_with_kernel_inject=True)
    output = pipe('Input String', use_cache=False)
    print(f"---device:{local_rank}---\npeak mem: {torch.cuda.max_memory_allocated() / 1024**3:.3f}")
    comm.barrier()
    utils.logger(f"output: {output}", local_rank)