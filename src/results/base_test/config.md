```python
class Config:
    num_layers = 14
    num_experts_per_tok = 2
    hidden_size = 1024
    num_experts = 16
    intermediate_size = 3584
    num_attention_heads = 16
    num_key_value_heads = 4
    head_dim = 64
    rms_norm_eps = 1e-5
    vocab_size = 32768
    token_router = TokenRouter.uniform
```