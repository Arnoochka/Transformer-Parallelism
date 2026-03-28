import torch
from torch import LongTensor, Tensor
from torch.nn import Module
from typing import List, Dict
from mytransformers.parallel.pipeline_parallel.pipeline.utils import MBatch
from mytransformers.parallel.pipeline_parallel_1.layers import FakeTupleSeqModule, FakeSeqModule
from transformers.cache_utils import DynamicCache
from mytransformers import utils

class GenerationFunc:
    @staticmethod
    def simple_generate(model: Module,
                        batches: List[Dict],
                        max_new_tokens: int,
                        eos_token_id: int,
                        pad_token_id: int,
                        use_cache: bool = False):
        for module in model.modules():
            if isinstance(module, FakeTupleSeqModule) or isinstance(module, FakeSeqModule):
                module.reset()
        inputs_ids = []
        attention_masks = []
        unfinished_sequences = []
        for batch in batches:
            ids = batch['input_ids']
            attn_mask = batch['attention_mask']
            inputs_ids.append(ids)
            attention_masks.append(attn_mask)
            unfinished_sequences.append(ids.new(ids.shape[0]).fill_(1))

        outputs = [None] * len(batches)

        for step in range(max_new_tokens):
            for idx, batch in enumerate(batches):
                if use_cache:
                    if step > 0:
                        past_key_values = outputs[idx].past_key_values
                        ids = inputs_ids[idx][:, -1:]
                    else:
                        past_key_values = DynamicCache()
                        ids = inputs_ids[idx]
                else:
                    past_key_values = None
                    ids = inputs_ids[idx]
                with torch.no_grad():
                    outputs[idx] = model(
                        input_ids=ids,
                        attention_mask=attention_masks[idx],
                        past_key_values=past_key_values,
                        use_cache=use_cache
                    )

            for idx in range(len(batches)):
                next_token_logits = outputs[idx].logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                eos_in_sents = next_token == eos_token_id
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
    
    
    @staticmethod
    def encode_generate(model: Module,
                        batches: List,
                        *args) -> List:
        for idx, batch in enumerate(batches):
            batch[idx] = model(**batch)
            
        return batches
    
    @staticmethod
    def simple_generate_encdec(model: Module,
                               batches: List,
                               max_new_tokens: int,
                               eos_token_id: int,
                               pad_token_id: int,
                               use_cache: bool = False):

        for batch_idx, batch in enumerate(batches):

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            encoder_outputs = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            encoder_hidden_states = encoder_outputs.last_hidden_state

            decoder_input_ids = torch.full(
                (input_ids.shape[0], 1),
                model.config.decoder_start_token_id,
                device=input_ids.device
            )

            unfinished_sequences = decoder_input_ids.new(decoder_input_ids.shape[0]).fill_(1)
            past_key_values = DynamicCache() if use_cache else None

            for step in range(max_new_tokens):

                if use_cache and step > 0:
                    decoder_inputs = decoder_input_ids[:, -1:]
                else:
                    decoder_inputs = decoder_input_ids

                outputs = model.decoder(
                    input_ids=decoder_inputs,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache
                )

                hidden_states = outputs.last_hidden_state
                logits = model.lm_head(hidden_states)
                past_key_values = outputs.past_key_values

                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                eos_in_sents = next_token == eos_token_id
                unfinished_sequences = unfinished_sequences.mul((~eos_in_sents).long())
                next_token = next_token * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                next_token = next_token.unsqueeze(-1)
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

                if unfinished_sequences.max() == 0:
                    break

            batches[batch_idx] = decoder_input_ids

        return decoder_input_ids
    
    @staticmethod
    def pipeline_generate_encdec(model: Module,
                                 batches: List[MBatch],
                                 max_new_tokens: int,
                                 eos_token_id: int,
                                 pad_token_id: int,
                                 use_cache: bool = False) -> List:

        unfinished_sequences = []
        input_ids_list = []
        attention_masks_list = []
        encoder_outputs_list = []

        for idx, mbatch in enumerate(batches):
            ids = mbatch.data['input_ids']
            attn_mask = mbatch.data['attention_mask']
            input_ids_list.append(ids)
            unfinished_sequences.append(ids.new(ids.shape[0]).fill_(1))
            attention_masks_list.append(attn_mask)

            enc_out = model.encoder(
                input_ids=ids,
                attention_mask=attn_mask
            )
            encoder_outputs_list.append(enc_out.last_hidden_state)

            input_ids_list[idx] = torch.full(
                (ids.shape[0], 1),
                model.config.decoder_start_token_id,
                device=ids.device
            )
        past_key_values_list = [DynamicCache() if use_cache else None for _ in batches]

        for step in range(max_new_tokens):
            for idx, mbatch in enumerate(batches):
                if use_cache and step > 0:
                    decoder_input_ids = input_ids_list[idx][:, -1:]
                else:
                    decoder_input_ids = input_ids_list[idx]

                mbatch.data = {
                    "input_ids": decoder_input_ids,
                    "attention_mask": attention_masks_list[idx],
                    "encoder_hidden_states": encoder_outputs_list[idx],
                    "past_key_values": past_key_values_list[idx]
                }
            outputs: List[MBatch] = model(batches, use_cache=use_cache)

            for idx, out in enumerate(outputs):
                logits = out.data['logits']
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                eos_in_sents = (next_token == eos_token_id)
                unfinished_sequences[idx] = unfinished_sequences[idx].mul((~eos_in_sents).long())
                next_token = next_token * unfinished_sequences[idx] + pad_token_id * (1 - unfinished_sequences[idx])
                next_token = next_token.unsqueeze(-1)

                input_ids_list[idx] = torch.cat([input_ids_list[idx], next_token], dim=-1)

                new_mask_val = unfinished_sequences[idx].unsqueeze(-1)
                attention_masks_list[idx] = torch.cat([attention_masks_list[idx], new_mask_val], dim=-1)

                if use_cache:
                    past_key_values_list[idx] = out.data['past_key_values']

            if torch.cat(unfinished_sequences).max() == 0:
                print(f"Все предложения завершены на шаге {step+1}")
                break

        return input_ids_list