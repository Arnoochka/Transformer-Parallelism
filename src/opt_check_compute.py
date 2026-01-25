import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from collections import OrderedDict

model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

all_activations = []

def create_hook(name, step_dict):
    def hook(module, input, output):
        if isinstance(output, tuple):
            shapes = []
            for o in output:
                if isinstance(o, torch.Tensor):
                    shapes.append(o.shape)
                elif isinstance(o, DynamicCache):
                    # Вместо shape выводим количество сохраненных слоев в кеше
                    shapes.append(f"DynamicCache(layers={len(o.key_cache)})")
                elif o is None:
                    shapes.append("None")
                else:
                    shapes.append(type(o).__name__)
            step_dict[name] = tuple(shapes)
        else:
            step_dict[name] = output.shape if output is not None else "None"
    return hook

# Prompt
prompt = ["Hello, my favorite model" for _ in range(2)] # Уменьшил для краткости вывода
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Инициализируем кеш и начальные входные данные
past_key_values = None
generated = input_ids
max_new_tokens = 5

for step in range(max_new_tokens):
    step_activations = OrderedDict()
    hooks = []
    
    # Регистрация хуков (остается без изменений)
    hooks.append(model.model.decoder.embed_tokens.register_forward_hook(create_hook("embedding", step_activations)))
    for i, block in enumerate(model.model.decoder.layers):
        hooks.append(block.self_attn.register_forward_hook(create_hook(f"block_{i}_self_attn", step_activations)))
        hooks.append(block.fc1.register_forward_hook(create_hook(f"block_{i}_fc1", step_activations)))

    with torch.no_grad():
        # ВАЖНО: передаем past_key_values и используем только последний токен после первого шага
        if past_key_values is None:
            model_inputs = {"input_ids": generated, "attention_mask": attention_mask}
        else:
            # Передаем только последний сгенерированный токен
            model_inputs = {
                "input_ids": generated[:, -1:], 
                "past_key_values": past_key_values,
                "attention_mask": attention_mask
            }

        outputs = model(**model_inputs, use_cache=False)
        
        # Обновляем кеш и маску для следующего шага
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Обновляем входные данные
        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long)], dim=-1)

    all_activations.append(step_activations)
    for h in hooks: h.remove()

# Вывод результатов
for step_idx, step_dict in enumerate(all_activations):
    print(f"\n=== Step {step_idx + 1} ===")
    for name, shape in step_dict.items():
        print(f"{name:30s}: {shape}")

print("\nGenerated text:", tokenizer.decode(generated[0], skip_special_tokens=True))