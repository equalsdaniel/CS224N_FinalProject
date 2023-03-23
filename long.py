# import time, random, numpy as np, argparse, sys, re, os
# from types import SimpleNamespace
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
#
# from bert import BertModel
# from classifier import BertSentimentClassifier, SentimentDataset
# from optimizer import AdamW
# from tqdm import tqdm

from dfp_datasets import load_multitask_data, load_multitask_test_data

# BertModel max seq len is 512: from tokenizer.py,
# PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
#     "bert-base-uncased": 512
# }

"""
1. Find better dataset if possible so don't have to config 
    - for if we don't want to truncate: https://huggingface.co/datasets/scientific_papers/viewer/pubmed/train
    - if we're cool truncating
    
2. Set up model using miniBERT embeds
3. make rouge scoring func

can we change other stuff in classifier? if yea, add a load state dict for model from .pt to line 267

Preproccess
    - First try truncating
    - If doesn't work, split into paragraphs of <= 512 tokens and apply again if need
    
"""
# from datasets import load_dataset
#
# data = load_dataset('tomasg25/scientific_lay_summarisation')
# print(data)

import time
import pandas as pd
import csv
import ast
import torch
from dfp_datasets import SentencePairDataset, SentenceClassificationDataset, SentenceClassificationTestDataset
from multitask_classifier import save_model, train_multitask, MultitaskBERT, get_args, seed_everything
from evaluation import model_eval_sst, test_model_multitask
from torch.utils.data import DataLoader


def get_laysum():
    train_ds = pd.read_json('laysum_dummy_data/elife_ds1/train.json').to_csv('laysum_dummy_data/elife_train.csv', encoding='utf-8')
    dev_ds = pd.read_json('laysum_dummy_data/elife_ds1/val.json').to_csv('laysum_dummy_data/elife_dev.csv', encoding='utf-8')
    test_ds = pd.read_json('laysum_dummy_data/elife_ds1/test.json').to_csv('laysum_dummy_data/elife_test.csv', encoding='utf-8')

    dev_sent_data = []
    dev_pair_data = []
    dev_ids = []
    with open('laysum_dummy_data/elife_dev.csv', 'r') as file:
        for record in csv.DictReader(file):
            flatten = [item for sublist in record for item in sublist]
            dev_full_texts = ''.join(flatten)
            dev_sent_data.append(dev_full_texts)
            dev_sums = ''.join(record['abstract'])
            dev_pair_data.append((dev_full_texts, dev_sums))
            dev_ids.append(record['id'])

    test_sent_data = []
    test_pair_data = []
    test_ids = []
    with open('laysum_dummy_data/elife_test.csv', 'r') as file:
        for record in csv.DictReader(file):
            flatten = [item for sublist in record for item in sublist]
            test_full_texts = ''.join(flatten)
            test_sent_data.append(test_full_texts)
            test_sums = ''.join(record['abstract'])
            test_pair_data.append((test_full_texts, test_sums))
            test_ids.append(record['id'])

    return dev_sent_data, dev_pair_data, dev_ids, test_sent_data, test_pair_data, test_ids


def test_laysum_multitask(args, model, device):
    dev_sent_data, dev_pair_data, dev_ids, test_sent_data, test_pair_data, test_ids = get_laysum()

    test_sent_data = SentenceClassificationTestDataset(test_sent_data, args)
    dev_sent_data = SentenceClassificationDataset(dev_sent_data, args)

    sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sst_test_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_test_data = SentencePairTestDataset(para_test_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_test_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    sts_test_data = SentencePairTestDataset(sts_test_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sts_test_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, dev_sts_corr, \
        dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                para_dev_dataloader,
                                                                sts_dev_dataloader, model, device)

    test_para_y_pred, test_para_sent_ids, test_sst_y_pred, \
        test_sst_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
        model_eval_test_multitask(sst_test_dataloader,
                                  para_test_dataloader,
                                  sts_test_dataloader, model, device)

    with open(args.sst_dev_out, "w+") as f:
        print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
        f.write(f"id \t Predicted_Sentiment \n")
        for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
            f.write(f"{p} , {s} \n")

    with open(args.sst_test_out, "w+") as f:
        f.write(f"id \t Predicted_Sentiment \n")
        for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
            f.write(f"{p} , {s} \n")

    with open(args.para_dev_out, "w+") as f:
        print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p} , {s} \n")

    with open(args.para_test_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(test_para_sent_ids, test_para_y_pred):
            f.write(f"{p} , {s} \n")

    with open(args.sts_dev_out, "w+") as f:
        print(f"dev sts corr :: {dev_sts_corr :.3f}")
        f.write(f"id \t Predicted_Similiary \n")
        for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
            f.write(f"{p} , {s} \n")

    with open(args.sts_test_out, "w+") as f:
        f.write(f"id \t Predicted_Similiary \n")
        for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
            f.write(f"{p} , {s} \n")

def test_model_laysum(args):
    with torch.no_grad():
        #sss
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)



# laysum_sentiment = []
#
# sentiment_data = []
# with open(train, 'r') as fp:
#     for record in csv.DictReader(fp,delimiter = '\t'):
#         sent = record['sentence'].lower().strip()
#         sentiment_data.append(sent)
#
# print(f"Loaded {len(sentiment_data)} test examples from {sentiment_filename}")
#
# paraphrase_data = []
# with open(paraphrase_filename, 'r') as fp:
#     for record in csv.DictReader(fp,delimiter = '\t'):
#         #if record['split'] != split:
#         #    continue
#         paraphrase_data.append((preprocess_string(record['sentence1']),
#                                 preprocess_string(record['sentence2']),
#                                 ))
#
# print(f"Loaded {len(paraphrase_data)} test examples from {paraphrase_filename}")
#
# similarity_data = []
# with open(similarity_filename, 'r') as fp:
#     for record in csv.DictReader(fp,delimiter = '\t'):
#         similarity_data.append((preprocess_string(record['sentence1']),
#                                 preprocess_string(record['sentence2']),
#                                 ))
#
# print(f"Loaded {len(similarity_data)} test examples from {similarity_filename}")
#
# return sentiment_data, paraphrase_data, similarity_data

