#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import string
from string import digits
import torch
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.utils.data import Dataset
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn.utils.rnn import pad_sequence

def read_file_to_list(path):
    with open(path, encoding="utf8") as f:
        content_list = f.readlines()
    content_list = [x.strip() for x in content_list]
    return content_list


def preprocess(data,src_lang_name,tgt_lang_name):
    # preProcess the data
    data[src_lang_name] = data[src_lang_name].apply(lambda x: re.sub("'", '', x).lower())
    data[tgt_lang_name] = data[tgt_lang_name].apply(lambda x: re.sub("'", '', x).lower())

    # remove special chars
    exclude = set(string.punctuation)  # set of all special chars
    # remove all the special chars
    data[src_lang_name] = data[src_lang_name].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    data[tgt_lang_name] = data[tgt_lang_name].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

    remove_digits = str.maketrans('', '', digits)
    data[src_lang_name] = data[src_lang_name].apply(lambda x: x.translate(remove_digits))
    data[tgt_lang_name] = data[tgt_lang_name].apply(lambda x: x.translate(remove_digits))

    data[tgt_lang_name] = data[tgt_lang_name].apply(lambda x: x.translate(remove_digits))

    # Remove extra spaces
    data[src_lang_name] = data[src_lang_name].apply(lambda x: x.strip())
    data[tgt_lang_name] = data[tgt_lang_name].apply(lambda x: x.strip())
    data[src_lang_name] = data[src_lang_name].apply(lambda x: re.sub(" +", " ", x))
    data[tgt_lang_name] = data[tgt_lang_name].apply(lambda x: re.sub(" +", " ", x))
    return data

def get_data(src_file_path, tgt_file_path,src_lang_name, tgt_lang_name, val_frac=0.01):
    raw_src = read_file_to_list(src_file_path)
    raw_tgt = read_file_to_list(tgt_file_path)
    data = pd.DataFrame({src_lang_name: raw_src, tgt_lang_name: raw_tgt})

    data =preprocess(data,src_lang_name,tgt_lang_name)

    val_split_idx = int(len(data) * val_frac)  # index on which to split
    data_idx = list(range(len(data)))  # create a list of ints till len of data
    np.random.shuffle(data_idx)
    train_idx = data_idx[:val_split_idx]
    val_idx = data_idx[val_split_idx:]
    train_data = data.iloc[train_idx].reset_index().drop('index', axis=1)
    val_data = data.iloc[val_idx].reset_index().drop('index', axis=1)
    return train_data,val_data


# In[8]:


SPECIAL_CHAR = {'<UNX>': 0, '<SOS>': 1, '<EOS>': 2, '<PAD>': 3}


# In[9]:


class Vocab:
    def __init__(self, max_size=10000, min_frequency=0):
        self.max_size = max_size
        self.min_frequency = min_frequency
        self.itos = {0: '<UNX>', 1: '<SOS>', 2: '<EOS>', 3: '<PAD>'}
        self.stoi = {j: i for i, j in self.itos.items()}

    def tokenize(self, sentance):
        return sentance.strip().split(' ')

    def build_vocab(self, sentance_list):
        freq = {}
        idx = 4
        for sentance in sentance_list:
            # print(sentance)
            # print(self.tokenize(sentance))
            for word in self.tokenize(sentance):
                # print(word)
                if word in freq.keys():
                    freq[word] += 1
                else:
                    freq[word] = 1
        # print(freq)
        # print("##########")
        freq = {k: v for k, v in freq.items() if v >= self.min_frequency}
        # print(freq)
        freq = dict(sorted(freq.items(), key=lambda x: -x[1])[:self.max_size - idx])
        # print(freq)
        for i in freq:
            self.itos[idx] = i
            self.stoi[i] = idx
            idx += 1

    def numericalize(self, sentance, use_sos_n_eos=True, ):
        tokens = self.tokenize(sentance)
        number = []
        if use_sos_n_eos:
            number.append(1)
        for token in tokens:
            if token in self.stoi.keys():
                number.append(self.stoi[token])
            else:
                number.append(0)
        if use_sos_n_eos:
            number.append(2)
        return number

    def stringify(self, num_list):
        str_ret = []
        for i in num_list:
            if i in self.itos.keys():
                str_ret.append(self.itos[i])
        return str_ret






class CusDataset(Dataset):
    def __init__(self, lang1, lang2, lang1_vocab=None, lang2_vocab=None):

        self.lang1_vocab = Vocab(max_size=40000, min_frequency=1) if lang1_vocab is None else lang1_vocab
        self.lang2_vocab = Vocab(max_size=40000, min_frequency=1) if lang2_vocab is None else lang2_vocab
        self.lang1 = lang1
        self.lang2 = lang2
        if lang1_vocab is not None:
            self.lang1_vocab.build_vocab(lang1)
        if lang2_vocab is not None:
            self.lang2_vocab.build_vocab(lang2)

    def __getitem__(self, n):
        return torch.Tensor(self.lang1_vocab.numericalize(self.lang1[n])), torch.Tensor(
            self.lang2_vocab.numericalize(self.lang2[n]))

    def __len__(self):
        return len(self.lang2)


class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch_data):
        #         breakpoint()
        source_data = [item[0] for item in batch_data]
        target_data = [item[1] for item in batch_data]
        source = pad_sequence(source_data, batch_first=False, padding_value=self.pad_idx)
        target = pad_sequence(target_data, batch_first=False, padding_value=self.pad_idx)
        return source, target


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    # breakpoint()
    src_padding_mask = (src == SPECIAL_CHAR['<PAD>']).transpose(0, 1)
    tgt_padding_mask = (tgt == SPECIAL_CHAR['<PAD>']).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask



def padding_mask(batch,pad_idx):
    return ((batch==pad_idx)==0).type(torch.IntTensor)

def subseqent_mask(size):
    return (torch.triu(torch.ones((size,size)),diagonal=1)==0).type(torch.IntTensor)

def mask(batch,pad_idx,seq_dim):
    pad_mask=padding_mask(batch,pad_idx)
    subseq_mask=subseqent_mask(batch.size(seq_dim))
    return (pad_mask & subseq_mask).type(torch.IntTensor)

