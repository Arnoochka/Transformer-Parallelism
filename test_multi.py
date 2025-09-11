import os
import torch
import torch.distributed as dist
from mytransformers import tp
import mytransformers.layers as l
from mytransformers import TransformerType, FFNType, init_distributed, logger
from mytransformers import utils

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
    num_layers = 1
    vocab_size = 10000
    max_len = 256

def main():
    single = l.TransformerEncoderDecoderModel(Config)
    model = init_distributed(single, tp.TPTransformerEncoderDecoderModelGenerator)
    tp_group = utils.TP_GROUP
    local_rank = dist.get_rank(tp_group)
    print(f"group:{tp_group}")
    if local_rank == 0:
        x = torch.randint(low=0, high=Config.vocab_size - 1, size=(Config.seq_len, ), dtype=int).cuda(local_rank)
        encoder_output = torch.randint(low=0, high=Config.vocab_size - 1, size=(Config.seq_len, ), dtype=int).cuda(local_rank)
    else:
        x = torch.zeros(Config.seq_len, dtype=int).cuda(local_rank)
        encoder_output = torch.zeros(Config.seq_len, dtype=int).cuda(local_rank)
    
    logger("broadcast", local_rank)    
    dist.broadcast(x, src=0, group=tp_group)
    dist.broadcast(encoder_output, src=0, group=tp_group)
    x = list(x)
    encoder_output = list(encoder_output)
    logger("generate", local_rank)
    # result = model.generate(x, encoder_output)
    logger([module[0] for module in model.named_children()], local_rank)
    memory = torch.cuda.max_memory_allocated()
    print(f"---device:{local_rank }---\npeak allocated memory: {memory / 1024**3:.3f} GB")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
