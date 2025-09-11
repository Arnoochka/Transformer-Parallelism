import torch
from transformers import pipeline
import os

if __name__ == "__main__":
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    print(torch.__version__)
    print(torch.cuda.is_available())
    pipe = pipeline("text2text-generation",
                    model="google/t5-v1_1-small",
                    device=torch.device('cuda'),
                    torch_dtype=torch.float32)
    output = pipe('Input String')
    
