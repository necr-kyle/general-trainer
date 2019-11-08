from optim import ScheduledOptim
from dataset import get_tutor_dataset

import argparse
import time
import torch
import json
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
import random
import copy
import torch.nn.functional as F

import logging
from collections import defaultdict
from transformers import (GPT2Tokenizer,
                        GPT2LMHeadModel,
                        GPT2Config)

logger = logging.getLogger(__name__)

def generate_sentence(model, max_len):
    start = torch.zeros((1, 24), dtype=torch.long)
    start[0][-1] = 5
    start[0][-2] = 9
    start[0][-3] = 6
    model.eval()
    with torch.no_grad():
        length = 1
        sentence = []
        output = model(start)[0]
        input = output.max(2)[1][0]
        sentence.append(input[-1])
        logger.debug(input)
        while sentence[-1] != 0 and length < max_len:
            input = input.unsqueeze(0)
            length += 1
            output = model(input)[0]
            input = output.max(2)[1][0]
            sentence.append(input[-1])
            logger.debug(input)
    print(sentence)
    return sentence

def test_language_generate_greedy(max_len=64):
    config = GPT2Config.from_pretrained("./models/gpt2/config.json")
    model = GPT2LMHeadModel.from_pretrained("./models/gpt2/pytorch_model.bin", config=config)
    token_list = [3,5,2]
    input = torch.tensor(token_list).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        for i in range(max_len):
            output = model(input)[0]
            output_id = int(output.max(2)[1][0,-1])
            token_list.append(output_id)
            input = torch.tensor(token_list[-24:]).unsqueeze(0)
            if output_id == 0:
                break
    print(token_list)

def gpt2_generate_beam_search(model, tokenizer, max_len=64, sentence=None, choice=5, keeps=10):
    if not sentence:
        sentence = "Peking University is the most famous university on Chinese website"
    token_list = tokenizer.encode(sentence)
    sentence = [token+' ' for token in sentence.split()]
    input = torch.tensor(token_list).unsqueeze(0)
    model.eval()
    candidates = [(token_list, 1.0)]
    with torch.no_grad():
        for i in range(max_len):
            new_candidates = []
            all_ends = 0
            for token_list, score in candidates:
                if token_list[-1] == 13:
                    new_candidates.append((token_list, score,))
                    all_ends += 1
                    continue
                input = torch.tensor(token_list).unsqueeze(0)
                output = model(input)[0]        # [1, seq_len, vocab_size]
                last_output = output[0, -1, :]          # [vocab_size]
                last_output = F.softmax(last_output, dim=0)
                top_k = torch.topk(last_output, choice)   # [choice]
                prob = last_output[top_k[1]]
                for idx, id in enumerate(top_k[1]):
                    new_list = copy.deepcopy(token_list)
                    new_list.append(int(id))
                    new_score = score * float(prob[idx])
                    new_candidates.append((new_list, new_score,))
            if all_ends >= keeps:
                break
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:keeps]

    for token_list, score in candidates:
        print("-"*40)
        print(f"Score: {score}")
        print(f"Token list: {token_list}")
        sentence = list(map(lambda x: tokenizer.decode(x), token_list))
        print(f"Sentence: {''.join(sentence)}")

def test_gpt():
    config = GPT2Config.from_pretrained("/data/home/kylekhuang/models/gpt2/gpt2-config.json")
    model = GPT2LMHeadModel.from_pretrained("/data/home/kylekhuang/models/gpt2/gpt2-pytorch_model.bin", config=config)
    tokenizer = GPT2Tokenizer('/data/home/kylekhuang/models/gpt2/gpt2-vocab.json', "/data/home/kylekhuang/models/gpt2/gpt2-merges.txt")
    # logging.basicConfig(filename="default.txt", level=logging.DEBUG, filemode='w')
    # gpt2_generate_greedy(model, tokenizer, sentence=sys.argv[1])
    gpt2_generate_beam_search(model, tokenizer, sentence=sys.argv[1])

if __name__ == "__main__":
    test_gpt()
