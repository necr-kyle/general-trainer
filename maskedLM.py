from transformers import BertConfig, BertTokenizer
from transformers import BertForMaskedLM, BertModel
import argparse
import torch


def raw_test(args, config, tokenizer):
    raw_model = BertModel.from_pretrained(args.model_path, config=config)
    input_tensor = torch.tensor(tokenizer.encode(args.sentence)).unsqueeze(0)  # Batch size 1
    pooling_output = raw_model(input_tensor)[1]
    print(pooling_output.shape)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='../../models/bert-base-chinese/bert-base-chinese-config.json', type=str, required=False)
    parser.add_argument("--model_path", default='../../models/bert-base-chinese/bert-base-chinese-pytorch_model.bin', type=str, required=False)
    parser.add_argument("--vocab_path", default='../../models/bert-base-chinese/bert-base-chinese-vocab.txt', type=str, required=False)
    parser.add_argument("--sentence", default="不存在语料里的表达可能是真[MASK]识，而存在语料里面的表达也可[MASK]是假知识，更[MASK]用提普遍存在的模型偏见了", type=str, required=False)
    
    args = parser.parse_args()

    config = BertConfig.from_pretrained(args.config_path)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, )
    model = BertForMaskedLM.from_pretrained(args.model_path, config=config)

    # insert_test(args, config, tokenizer, model)
    raw_test(args, config, tokenizer)


if __name__ == "__main__":
    main()