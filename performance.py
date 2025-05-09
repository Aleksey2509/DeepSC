# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import gc
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
# from bert4keras.backend import keras
# from bert4keras.models import build_bert_model
# from bert4keras.tokenizers import Tokenizer
from w3lib.html import remove_tags
from transformers import BertTokenizer, BertModel

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type = int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type = str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type = str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type = str)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# using pre-trained model to compute the sentence similarity
class Similarity():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.model = BertModel.from_pretrained("bert-large-cased", device_map="cuda")

    def normalize_tensor(self, pooled):
        pooled_normalized = torch.div(pooled, torch.max(pooled, dim=-1, keepdim=True).values)
        pooled_normalized = torch.nn.functional.normalize(pooled_normalized, dim=-1)

        return pooled_normalized

    def compute_similarity(self, real, predicted):
        full_len = len(real)
        batch_amount = 10
        batch_size = int(full_len / batch_amount)
        matmul_res = 0
        for i in tqdm(range(batch_amount)):
            stop = min((i + 1) * batch_size, full_len)

            batch_real = real[i * batch_size : stop]
            batch_predicted = predicted[i * batch_size : stop]

            encoded_real = self.tokenizer(batch_real, return_tensors='pt', padding=True, truncation=True).to('cuda')
            encoded_predicted = self.tokenizer(batch_predicted, return_tensors='pt', padding=True, truncation=True).to('cuda')

            output_real = self.model(**encoded_real)
            output_predicted = self.model(**encoded_predicted)

            # pooled_real = output_real['pooler_output']
            pooled_real = output_real['last_hidden_state']
            pooled_real = torch.sum(pooled_real, dim=1)
            pooled_real_normalized = self.normalize_tensor(pooled_real)
            pooled_real_normalized = torch.unsqueeze(pooled_real_normalized, 1)

            # pooled_predicted = output_predicted['pooler_output']
            pooled_predicted = output_predicted['last_hidden_state']
            pooled_predicted = torch.sum(pooled_predicted, dim=1)
            pooled_predicted_normalized = self.normalize_tensor(pooled_predicted)
            pooled_predicted_normalized = torch.unsqueeze(pooled_predicted_normalized, 2)

            addent = torch.sum(torch.bmm(pooled_real_normalized, pooled_predicted_normalized))
            matmul_res += addent

        matmul_res = matmul_res / full_len
        gc.collect()
    
        return matmul_res.cpu()


def performance(args, SNR, net):
    similarity = Similarity()
    bleu_scores_arr = [
            BleuScore(1, 0, 0, 0),
            BleuScore(0, 1, 0, 0),
            BleuScore(0, 0, 1, 0),
            BleuScore(0, 0, 0, 1)
        ]

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    score2 = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:

                    sents = sents.to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)

            bleu_scores = [[] for i in range(len(bleu_scores_arr))]
            sim_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                # 1-gram
                for i in range(len(bleu_scores_arr)):
                    bleu_scores[i].append(bleu_scores_arr[i].compute_blue_score(sent1, sent2)) # 7*num_sent
                sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
                print(f'Finished one similarity, {sim_score}')
            bleu_score = np.array(bleu_scores)
            bleu_score = np.mean(bleu_score, axis=-1)
            score.append(bleu_score)

            sim_score = np.array(sim_score)
            score2.append(sim_score)

    score1 = np.mean(np.array(score), axis=0)
    score2 = np.mean(np.array(score2), axis=0)

    return score1, score2

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0,3,6,9,12,15,18]

    args.vocab_file = './data/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx

    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path)
    deepsc.load_state_dict(checkpoint)
    print('model load!')

    print(args.channel)
    bleu_score, sim_score = performance(args, SNR, deepsc)
    print(f"Bleu and sim scores for channel {args.channel}: {bleu_score}, {sim_score}")

