

# BertModel max seq len is 512: from tokenizer.py,
# PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
#     "bert-base-uncased": 512
# }


# from datasets import load_dataset
#
# data = load_dataset('tomasg25/scientific_lay_summarisation')
# print(data)

import time
import pandas as pd
import csv
import ast
import torch
from dfp_datasets import SentencePairDataset, SentenceClassificationDataset, SentenceClassificationTestDataset, SentencePairTestDataset
from multitask_classifier import save_model, train_multitask, MultitaskBERT, get_args, seed_everything
from evaluation import model_eval_sst, test_model_multitask, model_eval_test_multitask, model_eval_multitask
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

    test_pair_data = SentencePairTestDataset(test_pair_data, args)
    dev_pair_data = SentencePairDataset(dev_pair_data, args)

    test_sent_dataloader = DataLoader(test_sent_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=test_sent_data.collate_fn)
    dev_sent_dataloader = DataLoader(dev_sent_data, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_sent_data.collate_fn)

    test_pair_dataloader = DataLoader(test_pair_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=test_pair_data.collate_fn)
    dev_pair_dataloader = DataLoader(dev_pair_data, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=dev_pair_data.collate_fn)


    dev_paraphrase_accuracy, dev_para_y_pred, dev_ids, \
        dev_sentiment_accuracy, dev_sst_y_pred, dev_ids, dev_sts_corr, \
        dev_sts_y_pred, dev_ids = model_eval_multitask(dev_sent_dataloader,
                                                       dev_pair_dataloader,
                                                       dev_pair_dataloader, model, device)

    test_para_y_pred, test_ids, test_sst_y_pred, \
        test_ids, test_sts_y_pred, test_ids = \
        model_eval_test_multitask(test_sent_dataloader,
                                  test_pair_dataloader,
                                  test_pair_dataloader, model, device)

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

        test_laysum_multitask(args, model, device)



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

