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
    def test_generate(model: Module,
                      mbatches: List[MBatch],
                      max_new_tokens: int,
                      eos_token_id: int,
                      pad_token_id: int,
                      use_cache: bool = False) -> List:

        outputs: List[MBatch] = model(mbatches, use_cache=use_cache)
        torch.cuda.synchronize() 
        vocab_size = 10000
        device = torch.cuda.current_device()

        for _ in range(max_new_tokens):
            new_mbatches = []
            for idx, out in enumerate(outputs):
                prev_ids =  out.data['logits']
                batch_size = prev_ids.size(0)

                next_ids = torch.randint(
                    0, vocab_size,
                    (batch_size, 1),
                    dtype=torch.long,
                    device=device,
                )

                new_mbatches.append(MBatch(
                    data={
                        "input_ids": next_ids,
                        "past_key_values": out.data.get('past_key_values'),
                    },
                    idx=idx,
                    stream=torch.cuda.Stream(),
                    event=torch.cuda.Event(),
                ))

            mbatches = new_mbatches
            outputs = model(mbatches, use_cache=use_cache)
            torch.cuda.synchronize() 

        return outputs
    
    
    @staticmethod
    def test_generate_merge(model, mbatches, max_new_tokens, eos_token_id, pad_token_id, use_cache=False):
        outputs = model(mbatches, use_cache=use_cache)
        torch.cuda.synchronize()

        caches = [out.data['past_key_values'] for out in outputs]
        def _merge_kv_caches(caches: List[DynamicCache]) -> DynamicCache:
            if len(caches) == 1:
                return caches[0]

            num_layers = len(caches[0].layers)

            seq_lens = [c.layers[0].keys.size(-2) for c in caches]
            max_seq = max(seq_lens)

            merged = DynamicCache()
            for layer_idx in range(num_layers):
                k_parts, v_parts = [], []

                for c, s in zip(caches, seq_lens):
                    k = c.layers[layer_idx].keys    # [b, heads, seq, head_dim]
                    v = c.layers[layer_idx].values

                    if s < max_seq:
                        pad = max_seq - s
                        k = torch.nn.functional.pad(k, (0, 0, pad, 0))
                        v = torch.nn.functional.pad(v, (0, 0, pad, 0))

                    k_parts.append(k)
                    v_parts.append(v)

                merged.update(
                    torch.cat(k_parts, dim=0),
                    torch.cat(v_parts, dim=0),
                    layer_idx,
                )

            for c in caches:
                del c

            return merged

        merged_cache = _merge_kv_caches(caches)

        next_ids = torch.cat([
            out.data['logits'][:, -1:, :].argmax(dim=-1)
            for out in outputs
        ], dim=0)

        for _ in range(max_new_tokens):
            decode_mbatch = MBatch(
                data={
                    "input_ids": next_ids,
                    "past_key_values": merged_cache,
                },
                idx=0,
                stream=torch.cuda.Stream(),
                event=torch.cuda.Event(),
            )

            outputs = model([decode_mbatch], use_cache=True)
            torch.cuda.synchronize()

            out = outputs[0]
            merged_cache = out.data['past_key_values']
            next_ids = out.data['logits'][:, -1:, :].argmax(dim=-1)

        return outputs