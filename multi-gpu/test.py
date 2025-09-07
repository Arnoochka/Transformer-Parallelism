import os
import torch
import torch.distributed as dist
from mytransformers import tp
import mytransformers.layers as l
from mytransformers import TransformerType, FFNType
import copy
from torch import Tensor

class Config:
    hidden_size = 16
    ffn_dim = 32
    num_query_heads = 4
    num_kv_heads = 2
    qk_dim = 8
    v_dim = 8
    dropout = 0.0
    bias = True
    batch_size = 4
    seq_len = 5
    transformer_type = TransformerType.EncoderDecoder
    ffn_type = FFNType.FFN
    eps = 10**(-5)
    elementwise_affine = True
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    num_layers = 1
    vocab_size = 32
    max_len = 20
    
def logger(log: str, rank: int) -> None:
    if rank == 0:
        print(log)

def torch_round(tensor, decimals):
    multiplier = 10 ** decimals
    return torch.round(tensor * multiplier) / multiplier

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    tp_group = dist.new_group(ranks=[0, 1], backend="nccl")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    attn = l.TransformerEncoderDecoderModel(Config)
    torch.cuda.set_device(local_rank)
    if rank == 0:
        x = torch.randint(low=0, high=31, size=(Config.seq_len, ), dtype=int).cuda(local_rank)
        encoder_output = torch.randint(low=0, high=31, size=(Config.seq_len, ), dtype=int).cuda(local_rank)
    else:
        x = torch.zeros(Config.seq_len, dtype=int).cuda(local_rank)
        encoder_output = torch.zeros(Config.seq_len, dtype=int).cuda(local_rank)
    
    logger("broadcast", rank)    
    dist.broadcast(x, src=0, group=tp_group)
    dist.broadcast(encoder_output, src=0, group=tp_group)
    x = list(x)
    encoder_output = list(encoder_output)
    logger("single", rank)
    x_1 = torch.empty(1)
    x_2 = torch.empty(1)
    if rank == 0:  
        model = attn.cuda(local_rank)
        with torch.no_grad():
            x_1 = Tensor(model.generate(copy.deepcopy(encoder_output), copy.deepcopy(x)))
    logger(f"x_1:\n{x_1}", rank)
    logger("multi generator", rank)
    tp.ParallelTransformerEncoderDecoderModelGenerator.config = Config
    model = tp.ParallelTransformerEncoderDecoderModelGenerator(attn, tp_group)
    logger("multi", rank)
    x_2 = Tensor(model.generate(copy.deepcopy(encoder_output), copy.deepcopy(x)))
    logger(f"x_2:\n{x_2}", rank)
    logger(f"is equal:{torch.all(torch_round(x_1, 3) == torch_round(x_2, 3))}", rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
