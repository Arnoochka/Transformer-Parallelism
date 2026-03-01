from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = [Image.open(requests.get(url, stream=True).raw) for _ in range(8)]

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')
model = AutoModel.from_pretrained('facebook/dinov2-giant')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape)
print(model)
