```python
class Config:
    num_layers = 14
    num_experts_per_tok = num_experts_per_tok
    hidden_size = 1024
    num_experts = num_experts
    intermediate_size = int(hidden_size * 2.5)
    head_dim = 64
    num_attention_heads = hidden_size // head_dim
    num_key_value_heads = num_attention_heads // 4 
    rms_norm_eps = 1e-5
    vocab_size = 32768
    token_router = TokenRouter.uniform
```