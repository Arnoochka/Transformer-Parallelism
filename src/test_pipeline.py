import os
import torch
from mytransformers import utils
from mytransformers import pp_custom
from mytransformers.parallel import pp
from transformers import (AutoTokenizer, OPTForCausalLM)
from mytransformers.benchmark import init_global_tracker
import pandas as pd
text = """В тихом городке на берегу озера жила старая библиотекарша по имени Элина. Каждый день она открывала двери своей маленькой библиотеки рано утром и оставалась там до самого заката, словно охраняя целую вселенную знаний. В библиотеке Элины было множество старинных книг с пожелтевшими страницами, каждая из которых хранила свои тайны. Иногда казалось, что книги шепчут друг другу, обмениваясь историями, которые не дожили до слуха обычного человека. Элина любила наблюдать за посетителями. Среди них были студенты, ищущие вдохновение, пожилые люди, которые возвращались к воспоминаниям детства, и дети, которым казалось, что книги оживают прямо на их глазах. Особым секретом библиотеки была маленькая комната в глубине здания, куда не каждый мог попасть. Там стоял старый глобус, на котором иногда появлялись странные светящиеся отметки. Элина шептала, что это секретное место хранит путь к забытым мирам и легендам. Однажды в библиотеку зашла девочка по имени Лия, которая только что переехала в город. Она была тихой и застенчивой, но в глазах ее горел огонь любопытства. Элина сразу заметила это. Подойдя к Лии, она протянула ей книгу с кожаным переплетом. «Эта книга найдет тебя, если ты готова к приключению», — сказала она таинственно. Лия взяла книгу и почувствовала, как страницы слегка дрожат в руках. В тот момент она поняла, что библиотека — это не просто место для чтения, а портал в миры, о которых она никогда не слышала. С этого дня Лия начала открывать для себя тайны, о которых никто не рассказывал. Каждая книга превращалась в дверь, за которой скрывались новые истории и невероятные приключения. Она встречала магов и ученых, путешествовала в страны, где ночь была ярче дня, и слышала истории, которые казались невозможными. Но главное, Лия поняла: настоящие приключения начинаются там, где начинается воображение."""

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    utils.init_distributed_cuda()
    first_stage = [utils.create_group([0]), [0]]
    second_stage = [utils.create_group([1]), [1]]
    pp_custom.OPTGenerator(module=model,
                           num_stages=2,
                           groups_info=[first_stage, second_stage],
                           final_group=first_stage[0],
                           embed_size=2048,
                           vocab_size=50272,
                           device=torch.cuda.current_device())
    utils.Logger.log_all_device(model)
    device = torch.cuda.current_device()
    texts =[text for _ in range(32)]
    inputs = tokenizer(texts, return_tensors="pt", max_length=256).to(device)
    inputs['use_cache'] = False
    input_ids = [pp.Microbatch(data=inputs,
                               idx=k,
                               stream=torch.cuda.Stream(),
                               event=torch.cuda.Event())
                 for k in range(1)]
    TRACKER = init_global_tracker()
    TRACKER.start()
    model(input_ids)
    df = TRACKER.stop()
    df.to_csv("one-batch-results.csv")
    utils.Logger.log_all_device(f"MODEL MEMORY: {utils.get_model_size(model):.3f}")
    
    