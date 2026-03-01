from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import requests

model_name = "facebook/dinov2-giant"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
).to(device)

image = [Image.open(requests.get(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
    stream=True
).raw).convert("RGB") for _ in range(8)]

inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.shape, inputs.pixel_values.shape)
print(f"{torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
print(model)