import torch
from transformers import AutoTokenizer, OPTForCausalLM 

model_name = "facebook/opt-350m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)

device = torch.device('cuda')
model.to(device)
model.eval()

texts = [
    "Deep learning is",
    "The meaning of life is",
    "Python is a programming language that"
]

inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40)

for i, text in enumerate(tokenizer.batch_decode(out, skip_special_tokens=True)):
    print(f"=== Sample {i} ===")
    print(text)
