from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
).to("cuda")

text = "translate English to German: Machine learning is amazing."

inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print(f"{torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
print(model)