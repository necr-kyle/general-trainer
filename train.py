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

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  BertForMaskedLM,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  RobertaForMaskedLM,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer,
                                  GPT2Tokenizer,
                                  GPT2LMHeadModel,
                                  GPT2Config)

logger = logging.getLogger(__name__)
info_level_dict = defaultdict(lambda : logging.INFO, {'debug': logging.DEBUG,
                                    'info': logging.INFO,
                                    'warning': logging.WARNING,
                                    'error': logging.ERROR,
                                    'critical': logging.CRITICAL})

model_dict = {'bert': (BertConfig, BertForMaskedLM),
              'roberta': (RobertaConfig, RobertaForMaskedLM),
              'gpt2': (GPT2Config, GPT2LMHeadModel)}

def draw_loss_curve(args, info_list):

    x = list(range(len(info_list[0]["loss_list"])))

    color_list = ['r-', 'b-', 'g-', 'y-', 'm-', 'k-', 'c-']
    for index, info in enumerate(info_list):
        plt.figure()
        plt.plot(x, info["loss_list"], color_list[index], 
                 label=f'{info["config"][0]}-{info["config"][1]} mode')
        plt.ylabel("loss")
        plt.xlabel(f'*{args.log_interval} batches')
        plt.legend(loc='upper right', shadow=True, fontsize='medium')

    plt.savefig('comparison.svg', format='svg')

def eval(args, model, eval_iter):
    model.eval()
    with torch.no_grad():
        logger.info('')
        logger.info('*' * 6 + '  Evaluation starts  ' + "*" * 6)
        start = time.time()
        running_loss = 0

        for inputs, targets in eval_iter:
            outputs = model(inputs, labels=targets)
            loss = outputs[0]
            logger.debug(f'model outputs: {outputs}')
            running_loss += loss.item()

        size = eval_iter.data_size()            
        end = time.time()
        logger.info('\t[Eval] avg loss: %.6f, time: %d seconds' 
                    % (running_loss/size*args.batch_size, end-start))
        logger.info('*' * 7 + '  Evaluation Ends  ' + "*" * 7)
        logger.info('')
        return running_loss/size*args.batch_size


def train(args, model, train_iter, eval_iter=None):

    if not args.no_cuda:
        model = model.cuda(args.device_no)
        model.train()
    # model = torch.nn.DataParallel(model, device_ids=(0,1,2))
    # train_data = load_data('train.txt')

    optimizer = ScheduledOptim(
                    optim.Adam(filter(lambda x: x.requires_grad, model.parameters())),
                    args.learning_rate,
                    args.warmup_steps)

    loss_list = []
    eval_loss_list = []
    # with torch.cuda.device(device_num):
    batch_count = 0
    running_loss = 0
    # start = time.time()
    while batch_count < args.training_steps:
        for inputs, targets in train_iter:
            if batch_count >= args.training_steps:
                break
            # input is a masked sequence 
            # target contains original word on the masked position, other positions are filled with -1
            # e.g.
            # input:  [101, 2342, 6537, 104,   104,  4423]
            # target: [-1,  -1,   -1,   10281, 8213, -1]

            logger.debug(f'inputs: {inputs}')
            logger.debug(f'targets: {targets}')
            logger.debug(f'sum: {sum(inputs[:,0])}')
            batch_count += 1
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, labels=targets)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

            logger.debug(f'model outputs: {outputs}')

            running_loss += loss.item()

            # write loss log
            if batch_count % args.log_interval == 0:
                loss_list.append(loss.item())
                logger.info('Batch:%6d, loss: %.6f  [%s]' % \
                        (batch_count, running_loss/args.log_interval, time.strftime("%D %H:%M:%S")))
                running_loss = 0

            # save model & curve
            if batch_count % args.checkpoint_interval == 0 or \
                        (batch_count < args.warmup_steps and batch_count % int(args.checkpoint_interval / 10) == 0):
                if eval_iter is not None:
                    eval_loss = eval(args, model, eval_iter)
                    eval_loss_list.append(eval_loss)
                    if eval_loss < min(eval_loss_list) and args.save_best and not args.no_checkpoint:
                        path = os.path.join(args.checkpoint_save_path, "model_pool", 
                                            f"{args.model_type}-best")
                        if not os.path.exists(path):
                            os.makedirs(path)
                        model.save_pretrained(path)
                        logger.info('Best model saved in %s' % path)
                if not args.no_checkpoint:
                    path = os.path.join(args.checkpoint_save_path, "tmp", f"{args.model_type}-{batch_count}")
                    if not os.path.exists(path):
                        os.makedirs(path)
                    model.save_pretrained(path)
                    logger.info('Model saved in %s' % path)
                    curve_info = {"train_loss_list": loss_list,
                                    "eval_loss_list": eval_loss_list}
                    with open(path + f'/{args.model_type}-{batch_count}-loss.pkl', 'wb+') as file:
                        pickle.dump(curve_info, file)
    return loss_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='bert', type=str, required=False)

    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_path", default=None, type=str, required=False,
                        help="use this when reading from existing checkpoint")

    parser.add_argument("--device_no", default=0, type=int, required=False,
                        help="which gpu you want the script to run on")
    parser.add_argument("--no_cuda", action="store_true",
                        help="if you want to use cpu")

    parser.add_argument("--training_steps", default=100000, type=int, required=False,
                        help="number of batches set for pre-training.")
    parser.add_argument("--warmup_steps", default=5000, type=int, required=False)
    parser.add_argument("--batch_size", default=6, type=int, required=False)
    parser.add_argument("--learning_rate", default=8e-3, type=float, required=False)
                        
    parser.add_argument("--train_data_path", default=None, type=str, required=False)
    parser.add_argument("--train_data_size", default=-1, type=int, required=False)
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--eval_data_path", default=None, type=str, required=False)
    parser.add_argument("--eval_data_size", default=-1, type=int, required=False)
    parser.add_argument("--no_eval", action="store_true")

    parser.add_argument("--checkpoint_save_path", default=None, type=str, required=False,
                        help="the checkpoints will be saved in ${path}/tmp/")
    parser.add_argument("--checkpoint_interval", default=50000, type=int, required=False,
                        help="after this number of batches the model will save a checkpoint")
    parser.add_argument("--no_checkpoint", action="store_true")

    parser.add_argument("--log_interval", default=5000, type=int, required=False,
                        help="how many batches you want for every display message")
    parser.add_argument("--logging_output", default="default.out", type=str, required=False)
    parser.add_argument("--logging_level", default='info', type=str, required=False,
                        help="If you want to show debug information, use 'debug'.")

    parser.add_argument("--allow_os_error", action="store_true",
                        help="if allow os error, a new model will be automatically used if the checkpoint is not found.")
    parser.add_argument("--debugging", action="store_true",
                        help="entering debug mode, lowering logging level and etc..")
    parser.add_argument("--random_seed", default=233, type=int, required=False,
                        help="random seed")
    args = parser.parse_args()

    if args.model_type not in model_dict.keys():
        logger.error("--model_type not in model_dict. Please try another.")
        return
    if not args.no_checkpoint and args.checkpoint_save_path is None:
        raise OSError("Need checkpoint_save_path or no_checkpoint.")
    if not args.no_train and (args.train_data_path is None or args.train_data_path == ''):
        raise OSError("Need train_data_path or no_train")
    if not args.no_eval and (args.eval_data_path is None or args.eval_data_path == ''):
        raise OSError("Need eval_data_path or no_eval")
    if args.no_train and args.no_eval:
        raise ValueError("No training and no evaluating make a meaningless script.")

    if args.debugging:
        args.logging_level = "debug"

    Config = model_dict[args.model_type][0]
    Model = model_dict[args.model_type][1]
    
    logging.basicConfig(filename=args.logging_output, level=info_level_dict[args.logging_level])

    json_args = json.dumps(vars(args), indent=4)
    logger.info(f"Experiment setting:\n{json_args}")

    model_config = Config.from_pretrained(args.config_path)
    if args.model_path is None or args.model_path == "":
        mlm = Model(config=model_config)
    else:
        try:
            mlm = Model.from_pretrained(args.model_path, config=model_config)
        except (OSError, AssertionError) as e:
            if not args.allow_os_error:
                raise e
            else:
                logging.info(f"Loading error: didn't find checkpoint at {args.model_path}. Create from scratch.")
                mlm = Model(config=model_config)

    if not args.no_train:
        if args.train_data_size <= 0:
            args.train_data_size = None
        train_iter = get_tutor_dataset(args.train_data_path, args.batch_size)
    if not args.no_eval:
        if args.eval_data_size <= 0:
            args.eval_data_size = None
        eval_iter = get_tutor_dataset(args.eval_data_path, args.batch_size)

    logger.info(f"Data loaded." +
                f"{len(train_iter.dataset)} sequences in train dataset." if not args.no_train else "" +
                f"{len(eval_iter.dataset)} sequences in eval dataset" if not args.no_eval else "" 
                )
        
    if not args.no_train:
        if not args.no_eval:
            start = time.time()
            train(args, mlm, train_iter, eval_iter)
            end = time.time()
            logger.info(f"{end-start} seconds training with {args.model_type} pmodel.")
        else:
            start = time.time()
            train(args, mlm, train_iter)
            end = time.time()
            logger.info(f"{end-start} seconds training with {args.model_type} pmodel.")

if __name__ == '__main__':
    main()

