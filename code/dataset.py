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


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=False)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)


# In[24]:


# def masking(src,tgt,pad_idx):
#     src_mask=torch.zeros_like(src)
#     src_mask[src==pad_idx]=-inf
#     tgt_mask=torch.zeros_like(tgt)


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


# In[25]:


torch.triu(torch.ones(3, 3), diagonal=1)

# In[34]:


model = Seq2SeqTransformer(num_encoder_layers=3,
                           num_decoder_layers=3,
                           emb_size=256,
                           nhead=4,
                           src_vocab_size=40000,
                           tgt_vocab_size=40000,
                           dim_feedforward=256,
                           dropout=0.1,
                           )

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

model = model.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SPECIAL_CHAR['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# In[35]:


from torch.utils.data import DataLoader, Dataset

batch_size = 10
num_workers = 0

# In[36]:


q = 3.44
print(f"test {q}")


# In[37]:


def train(model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    train_loader = DataLoader(trainDataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, collate_fn=Collate(pad_idx=SPECIAL_CHAR['<PAD>']), pin_memory=True)
    for j, (src, tgt) in enumerate(train_loader):
        src = src.to(device=DEVICE)
        tgt = tgt.to(device=DEVICE)
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :].type(torch.LongTensor).to(device=DEVICE)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        src_mask = src_mask.to(device=DEVICE)
        tgt_mask = tgt_mask.to(device=DEVICE)
        src_padding_mask = src_padding_mask.to(device=DEVICE)
        tgt_padding_mask = tgt_padding_mask.to(device=DEVICE)

        predicted = model(src=src,
                          trg=tgt_input,
                          src_mask=src_mask,
                          tgt_mask=tgt_mask,
                          src_padding_mask=src_padding_mask,
                          tgt_padding_mask=tgt_padding_mask,
                          memory_key_padding_mask=src_padding_mask)

        #         breakpoint()

        optimizer.zero_grad()
        loss = loss_fn(predicted.reshape(-1, predicted.size()[-1]), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss

    return total_loss / len(train_loader)


def val(model):
    val_loader = DataLoader(valDataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=True, collate_fn=Collate(pad_idx=SPECIAL_CHAR['<PAD>']), pin_memory=True)
    model.eval()
    val_total_loss = 0
    for j, (src, tgt) in enumerate(val_loader):
        src = src.to(device=DEVICE)
        tgt = tgt.to(device=DEVICE)
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :].type(torch.LongTensor).to(device=DEVICE)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        src_mask = src_mask.to(device=DEVICE)
        tgt_mask = tgt_mask.to(device=DEVICE)
        src_padding_mask = src_padding_mask.to(device=DEVICE)
        tgt_padding_mask = tgt_padding_mask.to(device=DEVICE)

        predicted = model(src=src,
                          trg=tgt_input,
                          src_mask=src_mask,
                          tgt_mask=tgt_mask,
                          src_padding_mask=src_padding_mask,
                          tgt_padding_mask=tgt_padding_mask,
                          memory_key_padding_mask=src_padding_mask)

        #         breakpoint()

        val_loss = loss_fn(predicted.reshape(-1, predicted.size()[-1]), tgt_output.reshape(-1))
        val_total_loss += val_loss
    return val_total / len(val_loader)


# In[38]:


EPOCHS = 1
for i in range(EPOCHS):
    train_loss = train(model, loss_fn, optimizer)
    #     val_loss=val(model)
    val_loss = 0

    print(f"training loss ---- {train_loss} ::: testing loss ---- {val_loss}")


# In[39]:


def greedy_decode(src, src_mask, start_index, end_index, max_len, model):
    src = src.to(device=DEVICE)
    src_mask = src_mask.to(device=DEVICE)
    mem = model.encode(src, src_mask).to(device=DEVICE)
    ys = torch.Tensor([[start_index]]).type(torch.long).to(device=DEVICE)
    for i in range(max_len):
        tgt_mask = generate_square_subsequent_mask(len(ys)).type(torch.bool).to(DEVICE)
        pred = model.decode(ys, mem, tgt_mask)
        output = model.generator(pred)
        ys = torch.cat((ys, output))
        if output == stop_index:
            break

    return ys


def greedy_decode1(model, src, src_mask, max_len, start_symbol, end_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break
    return ys


# In[42]:


temp = "Comment allez vous"
temp = "what is your name"
temp = "Big people aren't always strong"
test_in = torch.Tensor(trainDataset.lang1_vocab.numericalize(temp)).to(device=DEVICE)
test_mask = torch.zeros(test_in.size()[0], test_in.size()[0]).to(device=DEVICE)
test_in = test_in.reshape(-1, 1)
print()

# greedy_decode(test_in,test_mask,SPECIAL_CHAR['<SOS>'],SPECIAL_CHAR['<EOS>'],50,model)
values = greedy_decode1(model, test_in, test_mask, 50, SPECIAL_CHAR['<SOS>'], SPECIAL_CHAR['<EOS>'])

# In[43]:


trainDataset.lang2_vocab.stringify(values.reshape(-1).to(dtype=torch.int32).cpu().numpy())

# In[96]:


values.reshape(-1).to(dtype=torch.int32)

# In[ ]:




