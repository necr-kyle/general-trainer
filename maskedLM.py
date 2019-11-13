from transformers import BertConfig, BertTokenizer
from transformers import BertForMaskedLM, BertModel
import argparse
import torch


def raw_test(args, config, tokenizer):
    raw_model = BertModel.from_pretrained("./bert-base-chinese/bert-base-chinese-pytorch_model.bin", config=config)
    input_tensor = torch.tensor(tokenizer.encode(args.sentence)).unsqueeze(0)  # Batch size 1
    pooling_output = raw_model(input_tensor)
    print(pooling_output[1].shape)


def mask_test(args):
    input_tensor = torch.tensor(tokenizer.encode(args.sentence)).unsqueeze(0)  # Batch size 1
    mask_pos = (input_tensor==103).to(torch.long).nonzero()
    print(mask_pos)
    outputs = model(input_tensor, masked_lm_labels=input_tensor)
    loss, prediction_scores = outputs[:2]
    print(prediction_scores[mask_pos])
    print(args.sentence)
    ids = prediction_scores.max(2)[1].squeeze(0)
    output_sentence = ''.join(list(map(lambda x: tokenizer.decode(x), ids.tolist())))
    print(output_sentence)

def insert_test(args, config, tokenizer, model):
    input_list = tokenizer.encode(args.sentence)
    input_tensor = torch.tensor(input_list).unsqueeze(0)  # Batch size 1

    mask_pos = (input_tensor==103).to(torch.long).nonzero()
    print(mask_pos)
    mask_pos = mask_pos.numpy().T

    outputs = model(input_tensor, masked_lm_labels=input_tensor)
    loss, prediction_scores = outputs[:2]
    needed_scores = prediction_scores[mask_pos]
    ids = needed_scores.max(1)[1]

    i = 0
    while 103 in input_list:
        index = mask_pos[1][i]
        input_list.insert(index, int(ids[i]))
        input_list.remove(103)
        i += 1

    print("Input:", args.sentence)
    output_sentence = ''.join([tokenizer.decode(x) for x in input_list])
    print("Output:", output_sentence)


def bert_wrapper(input_list):
    input_list.insert(0, 101)
    input_list.append(102)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=None, type=str, required=True)
    parser.add_argument("--model_path", default=None, type=str, required=True)
    parser.add_argument("--vocab_path", default=None, type=str, required=True)
    parser.add_argument("--sentence", type=str, required=True)
    
    args = parser.parse_args()

    config = BertConfig.from_pretrained(args.config)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertForMaskedLM.from_pretrained(args.model_path, config=config)

    insert_test(args, config, tokenizer, model)


if __name__ == "__main__":
    main()