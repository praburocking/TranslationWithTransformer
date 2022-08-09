# -*- coding: utf-8 -*-


import torchtext
import torchdata
from torchdata import datapipes as dp
from torch.utils.data import DataLoader
import torchdata.datapipes as dp
from torchtext.vocab import build_vocab_from_iterator
import numpy as np

from torchtext.vocab import vocab
from collections import Counter, OrderedDict

from collections import Counter,OrderedDict
from torchtext.data.utils import get_tokenizer

from typing import Iterable, List

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from copy import deepcopy
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


eng_data_path="../data/train_eng_trans.txt"
fr_data_path= "../data/train_french_trans.txt"

val_eng_data_path="../data/val_eng_trans.txt"
val_fr_data_path= "../data/val_french_trans.txt"

def read_data(src_data_path,tgt_data_path):
    fo1=dp.iter.FileOpener(datapipe= [src_data_path], encoding='utf-8').readlines().map(lambda x:x[1])
    fo2=dp.iter.FileOpener(datapipe= [tgt_data_path], encoding='utf-8').readlines().map(lambda x:x[1])
    fo=fo1.zip(fo2)
    return fo

def create_src_tgt_vocab(dl,src_tokenizer,tgt_tokenizer,src_specials,tgt_specials,src_max_tokens=250000,tgt_max_tokens=250000,
                         src_min_freq=1,tgt_min_freq=1,special_first=True,src_default='<unk>',tgt_default='<unk>'):
    src_counter=Counter()
    tgt_counter=Counter()
    
    for i,data in enumerate(dl):
      src_counter.update(src_tokenizer(data[0]))
      tgt_counter.update(tgt_tokenizer(data[1]))

   
    # First sort by descending frequency, then lexicographically
    src_sorted_by_freq_tuples = sorted(src_counter.items(), key=lambda x: (-x[1], x[0]))
    tgt_sorted_by_freq_tuples = sorted(tgt_counter.items(), key=lambda x: (-x[1], x[0]))

    """
    src  token processing
    """
    if src_max_tokens is None:
        src_ordered_dict = OrderedDict(src_sorted_by_freq_tuples)
        
    else:
        assert len(src_specials) < src_max_tokens, "len(specials) >= max_tokens, so the vocab will be entirely special tokens."
        src_ordered_dict = OrderedDict(src_sorted_by_freq_tuples[: src_max_tokens - len(src_specials)])

    """
    tgt  token processing
    """
    if tgt_max_tokens is None:
        tgt_ordered_dict = OrderedDict(tgt_sorted_by_freq_tuples)
        
    else:
        assert len(tgt_specials) < tgt_max_tokens, "len(specials) >= max_tokens, so the vocab will be entirely special tokens."
        tgt_ordered_dict = OrderedDict(tgt_sorted_by_freq_tuples[: tgt_max_tokens - len(tgt_specials)])

    src_word_vocab = vocab(src_ordered_dict, min_freq=src_min_freq, specials=src_specials, special_first=special_first)
    src_word_vocab.set_default_index(src_word_vocab[src_default])
    tgt_word_vocab = vocab(tgt_ordered_dict, min_freq=tgt_min_freq, specials=tgt_specials, special_first=special_first)
    tgt_word_vocab.set_default_index(tgt_word_vocab[tgt_default])
    return src_word_vocab,tgt_word_vocab
    """
    Save and retrieve the stats dict to save and load the vocab module again as it is a instance of nn.Module
    """


SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']
tok_en=get_tokenizer('spacy', language='en_core_web_sm')
tok_fr=get_tokenizer('spacy', language='fr_core_news_sm')
data=read_data(eng_data_path,fr_data_path)
src_vocab,tgt_vocab=create_src_tgt_vocab(iter(data),tok_en,tok_fr,SPECIAL_SYMBOLS,SPECIAL_SYMBOLS,src_max_tokens=250000,tgt_max_tokens=250000,
                                         src_min_freq=1,tgt_min_freq=1,special_first=True,src_default='<unk>',tgt_default='<unk>')
UNK,PAD_IDX,BOS_IDX,EOS_IDX=0,1,2,3


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
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

## Seq2Seq Network
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
                                       dropout=dropout)
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
        # print("src size --- "+str(src.size())+" :: tgt size--- "+str(trg.size())+" :: src_embedding --- "+str(src_emb.size())+" :: tgt embedding --- "+str(src_emb.size()))
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

"""During training, we need a subsequent word mask that will prevent model to look into
the future words when making predictions. We will also need masks to hide
source and target padding tokens. Below, let's define a function that will take care of both. 



"""

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt,pad_idx=PAD_IDX):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

"""Let's now define the parameters of our model and instantiate the same. Below, we also 
define our loss function which is the cross-entropy loss and the optmizer used for training.



"""

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(src_vocab.get_stoi())
TGT_VOCAB_SIZE = len(tgt_vocab.get_stoi())
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int],bos_idx=BOS_IDX,eos_idx=EOS_IDX):
    return torch.cat((torch.tensor([bos_idx]),
                      torch.tensor(token_ids), 
                      torch.tensor([eos_idx])))


src_text_transform = sequential_transforms(tok_en, #Tokenization
                                               src_vocab, #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor
tgt_text_transform = sequential_transforms(tok_fr, #Tokenization
                                               tgt_vocab, #Numericalization
                                               tensor_transform)


# function to collate data samples into batch tesors
def collate_fn(batch,pad_idx=PAD_IDX):
    
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_text_transform(src_sample.rstrip("\n")))
        tgt_batch.append(tgt_text_transform(tgt_sample.rstrip("\n")))
    # breakpoint()
    src_batch = pad_sequence(src_batch, padding_value=pad_idx,batch_first=False)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx,batch_first=False)
    return src_batch, tgt_batch

"""Let's define training and evaluation loop that will be called for each 
epoch.



"""

from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter=iter(data)
    train_dataloader = DataLoader(data, batch_size=BATCH_SIZE, collate_fn=collate_fn,drop_last=True)

    for i,(src, tgt) in enumerate(train_dataloader):
        # breakpoint()
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        if i%100==0:
            print("batch..... "+str(i))

    return losses

#
# def evaluate(model):
#     model.eval()
#     losses = 0
#
#     val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
#     val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
#
#     for src, tgt in val_dataloader:
#         src = src.to(DEVICE)
#         tgt = tgt.to(DEVICE)
#
#         tgt_input = tgt[:-1, :]
#
#         src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
#
#         logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
#
#         tgt_out = tgt[1:, :]
#         loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
#         losses += loss.item()
#
#     return losses / len(val_dataloader)


from timeit import default_timer as timer
NUM_EPOCHS = 1

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss=0
    # val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

states_dict = deepcopy(transformer.state_dict())
torch.save(states_dict, "transformer_param_torch")
transformer.load_state_dict(torch.load('transformer_param_torch'),strict=False)

# function to generate output sequence using greedy algorithm 
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
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
        if next_word == SPECIAL_SYMBOLS[EOS_IDX]:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = src_text_transform(src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(tgt_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

print(translate(transformer, "how are you doing ?"))
