import torch
import json
from torch.distributed import ProcessGroup
from typing import List, Optional, Union
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer
from .BenchmarkStats import BenchmarkStats
from .Timers import timers
import time



class BenchmarkModel:
    def __init__(self,
                 model_name: Optional[str],
                 model_getter: PreTrainedModel,
                 generator: torch.nn.Module,
                 tokenizer: AutoTokenizer,
                 config: PretrainedConfig,
                 loops: int = 1,
                 batch_size: int = 1,
                 description: Optional[str] = None,
                 tool: Optional[str] = None,
                 parallelism: Optional[Union[str, List[str]]] = None,
                 dtype: torch.dtype = torch.float32,
                 save_model_config: bool = True):
        
        self.model_name = model_name
        self.model = model_getter.from_pretrained(model_name,
                                                  config=config,
                                                  torch_dtype=dtype)
        self.generator = generator
        self.tokenizer = tokenizer.from_pretrained(model_name)
        self.loops = loops
        self.batch_size = batch_size
        self.description = description
        self.tool = tool
        self.parallelism = parallelism
        self.model_config_dir = None
        self.dtype = str(dtype).split(".")[-1]
        
        if save_model_config:
            with open(f"{model_name}.json", 'w', encoding='utf-8') as file:
                json.dump(config, file, ensure_ascii=False, indent=4)
            
        
    def __call__(self,
                 input_data: Union[str, List[str]],
                 group: ProcessGroup) -> BenchmarkStats:
        
        
        
        input = torch.LongTensor(self.tokenizer.encode('Input String'))\
            .unsqueeze(0).to(torch.cuda.current_device())
        