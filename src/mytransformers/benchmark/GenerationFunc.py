import torch
from torch import LongTensor, Tensor
from torch.nn import Module
from typing import List
from mytransformers.parallel.pipeline_parallel.pipeline.utils import MBatch
from transformers.cache_utils import DynamicCache
from mytransformers import utils

class GenerationFunc:
    
    @staticmethod
    def simple_generate(model: Module,
                        input_ids: LongTensor,
                        attention_mask: LongTensor,
                        max_new_tokens: int,
                        eos_token_id: int,
                        pad_token_id: int,
                        use_cache: bool = False):

        past_key_values = None
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        for step in range(max_new_tokens):
            if use_cache and step > 0:
                model_inputs = input_ids[:, -1:]
            else:
                model_inputs = input_ids

            outputs = model(
                input_ids=model_inputs,
                attention_mask=attention_mask,
                past_key_values=past_key_values, 
                use_cache=use_cache
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            eos_in_sents = next_token == eos_token_id
            unfinished_sequences = unfinished_sequences.mul((~eos_in_sents).long())
            next_token = next_token * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            next_token = next_token.unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            new_mask_values = unfinished_sequences.unsqueeze(-1)
            attention_mask = torch.cat([attention_mask, new_mask_values], dim=-1)


            if unfinished_sequences.max() == 0:
                print(f"Все предложения завершены на шаге {step+1}")
                break

        return input_ids
    
    @staticmethod
    def pipeline_generate(model: Module,
                          mbatches: List[MBatch],
                          max_new_tokens: int,
                          eos_token_id: int,
                          pad_token_id: int,
                          use_cache: bool = False) -> List:

        unfinished_sequences = []
        inputs_ids = []
        attention_masks = []
        for idx, mbatch in enumerate(mbatches):
            ids = mbatch.data['input_ids']
            attn_mask = mbatch.data['attention_mask']
            inputs_ids.append(ids)
            unfinished_sequences.append(ids.new(ids.shape[0]).fill_(1))
            attention_masks.append(attn_mask)

        for step in range(max_new_tokens):
            for idx, mbatch in enumerate(mbatches):
                if use_cache:
                    if step > 0:
                        past_key_values = outputs[idx].data['past_key_values']
                        ids = inputs_ids[idx][:, -1:]
                    else:
                        past_key_values = DynamicCache()
                        ids = inputs_ids[idx]
                else:
                    ids = inputs_ids[idx]
                    past_key_values = None  
                mbatches[idx].data = {
                    "input_ids": ids,
                    "attention_mask": attention_masks[idx],
                    "past_key_values": past_key_values
                }
            outputs: List[MBatch] = model(mbatches, use_cache=use_cache)
            for idx, out in enumerate(outputs):
                logits = out.data['logits']
                
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                eos_in_sents = (next_token == eos_token_id)
                unfinished_sequences[idx] = unfinished_sequences[idx].mul((~eos_in_sents).long())
                next_token = next_token * unfinished_sequences[idx] + pad_token_id * (1 - unfinished_sequences[idx])
                next_token = next_token.unsqueeze(-1)
                
                inputs_ids[idx] = torch.cat([inputs_ids[idx], next_token], dim=-1)

                new_mask_val = unfinished_sequences[idx].unsqueeze(-1)
                attention_masks[idx] = torch.cat([attention_masks[idx], new_mask_val], dim=-1)

            if torch.cat(unfinished_sequences).max() == 0:
                print(f"Все предложения завершены на шаге {step+1}")
                break

        return inputs_ids