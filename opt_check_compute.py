import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict
model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Словарь: step -> {layer_name -> shape}
all_activations = []

def create_hook(name, step_dict):
    def hook(module, input, output):
        if isinstance(output, tuple):
            step_dict[name] = tuple(o.shape if o is not None else "None" for o in output)
        else:
            step_dict[name] = output.shape if output is not None else "None"
    return hook

# Prompt
prompt = ["Hello, my favorite model" for _ in range(11)]
inputs = tokenizer(prompt, return_tensors="pt", max_length=64)
input_ids = inputs["input_ids"]

# Генерация пошагово
max_new_tokens = 5
generated = input_ids

for step in range(max_new_tokens):
    step_activations = OrderedDict()
    
    # Регистрируем хуки на каждый шаг
    hooks = []
    
    # Embeddings
    hooks.append(model.model.decoder.embed_tokens.register_forward_hook(create_hook("embedding", step_activations)))
    hooks.append(model.model.decoder.embed_positions.register_forward_hook(create_hook("positional_embedding", step_activations)))

    # Слои
    for i, block in enumerate(model.model.decoder.layers):
        hooks.append(block.self_attn.register_forward_hook(create_hook(f"block_{i}_self_attn", step_activations)))
        hooks.append(block.self_attn_layer_norm.register_forward_hook(create_hook(f"block_{i}_self_attn_ln", step_activations)))
        hooks.append(block.fc1.register_forward_hook(create_hook(f"block_{i}_fc1", step_activations)))
        hooks.append(block.activation_fn.register_forward_hook(create_hook(f"block_{i}_activation", step_activations)))
        hooks.append(block.fc2.register_forward_hook(create_hook(f"block_{i}_fc2", step_activations)))
        hooks.append(block.final_layer_norm.register_forward_hook(create_hook(f"block_{i}_final_ln", step_activations)))
    
    hooks.append(model.model.decoder.final_layer_norm.register_forward_hook(create_hook("decoder_final_ln", step_activations)))

    # Forward для последнего токена
    with torch.no_grad():
        outputs = model(input_ids=generated, use_cache=False)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    # Добавляем новый токен к последовательности
    generated = torch.cat([generated, next_token], dim=-1)
    
    # Сохраняем результаты шага
    all_activations.append(step_activations)
    
    # Убираем хуки
    for h in hooks:
        h.remove()

# Вывод результатов
for step_idx, step_dict in enumerate(all_activations):
    print(f"\n=== Step {step_idx + 1} ===")
    for name, shape in step_dict.items():
        print(f"{name:30s}: {shape}")

# Получаем сгенерированный текст
generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print("\nGenerated text:", generated_text)
