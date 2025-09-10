import os
import torch
import torch.distributed as dist
from mytransformers import tp
import mytransformers.layers as l
from mytransformers import TransformerType, FFNType

class Config:
    hidden_size = 512
    ffn_dim = 1024
    num_query_heads = 128
    num_kv_heads = 32
    qk_dim = 1024
    v_dim = 1024
    dropout = 0.0
    bias = True
    batch_size = 4
    seq_len = 64
    transformer_type = TransformerType.EncoderDecoder
    ffn_type = FFNType.FFN
    eps = 10**(-5)
    elementwise_affine = False
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    num_layers = 7
    vocab_size = 10000
    max_len = 256
    
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
    single = l.TransformerEncoderDecoderModel(Config)
    torch.cuda.set_device(local_rank)
    if rank == 0:
        x = torch.randint(low=0, high=Config.vocab_size - 1, size=(Config.seq_len, ), dtype=int).cuda(local_rank)
        encoder_output = torch.randint(low=0, high=Config.vocab_size - 1, size=(Config.seq_len, ), dtype=int).cuda(local_rank)
    else:
        x = torch.zeros(Config.seq_len, dtype=int).cuda(local_rank)
        encoder_output = torch.zeros(Config.seq_len, dtype=int).cuda(local_rank)
    
    logger("broadcast", rank)    
    dist.broadcast(x, src=0, group=tp_group)
    dist.broadcast(encoder_output, src=0, group=tp_group)
    x = list(x)
    encoder_output = list(encoder_output)
    logger("generator", rank)
    tp.TPTransformerEncoderDecoderModelGenerator.config = Config
    model = tp.TPTransformerEncoderDecoderModelGenerator(single, tp_group)
    logger("generate", rank)
    result = model.generate(x, encoder_output)
    memory = torch.cuda.max_memory_allocated()
    print(f"---device:{rank}---\npeak allocated memory: {memory / 1024**3:.3f} GB")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
