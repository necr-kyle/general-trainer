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
import pickle
import random

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

if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("./model/gpt2-best")
    logging.basicConfig(filename="default.txt", level=logging.DEBUG, filemode='w')
    generate_sentence(model, 96)