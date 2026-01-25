import torch
from transformers import (AutoTokenizer, OPTForCausalLM)
text = "Deepspeed is "
if __name__ == "__main__":
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    device = torch.cuda.current_device()
    model = model.to(device)
    inputs = tokenizer([text for _ in range(64)], return_tensors="pt", max_length=256).to(device)
    outputs = model.generate(**inputs, use_cache=True)
    print(outputs)
    
    