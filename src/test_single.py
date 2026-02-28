from transformers import AutoTokenizer, AutoModel, DebertaV2Model
import torch

model_name = "microsoft/deberta-v2-xlarge"
device = torch.cuda.current_device()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float32
).to(device)

inputs = tokenizer("Distributed inference is cool! fdgdfaghsdjfkj gredhgjdshaR HTDHA. rehtsjkjshg rghsags", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

print(outputs)