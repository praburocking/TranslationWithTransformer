import pandas as pd
from torchtext.vocab import vocab,build_vocab_from_iterator,Vocab
import spacy
from typing import Iterable, List
from torchtext.data.utils import get_tokenizer
from copy import deepcopy
import torch
from torchtext._torchtext import Vocab as VocabPybind
from collections import Counter,OrderedDict
import time
from torch.utils.data import DataLoader
from torchdata.datapipes import iter as dp_iter




# def get_data(data_path):
#     return pd.read_csv(data_path,encoding='utf-8',chunksize=1,iterator=True)
#
# en_nlp=spacy.blank('en')
# fr_nlp=spacy.blank('fr')
# def tokenize(tf_reader,tokenizer)-> List[str]:
#     for i,data in enumerate(tf_reader):
#         yield tokenizer(str(data['en'][i]))
#         # yield (data['en'][i]).split(' ')
#         if i==10000:
#             break
#         if i%100==0:
#             print("progress....."+str(i)+"/10000")
# data_path='R:\\studies\\transformers\\translation\\data\\en-fr.csv'
# tf_reader=get_data(data_path)
#
# seconds1 = time.time()
# UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
# tokenizer=get_tokenizer('spacy', language='en_core_web_sm')
# # en_vocab=build_vocab_from_iterator(tokenize(tf_reader), min_freq=1, specials=special_symbols,special_first=True)
# src_vocab=build_vocab_from_iterator(tokenize(tf_reader,tokenizer),specials=special_symbols)
#
# torch.save(src_vocab, "src_vocab.vocab")
# re_vocab=torch.load('src_vocab.vocab')
# # print(re_vocab.get_stoi())
# total_seconds=time.time()-seconds1
# print("total time taken in seconds ... "+str(total_seconds))


#########################



def get_data_pipes(file_path: str, mode: str = "train", batch_size: int = 5000):
    buffer_size = 10000
    dp = dp_iter.FileOpener([file_path], encoding='utf-8').parse_csv(skip_lines=1).shuffle(buffer_size=buffer_size)
    dp = dp.sharding_filter().batch(batch_size=batch_size, drop_last=True)
    return dp

def get_data_loader(data_pipe,batch_size:int=5000):
   return DataLoader(data_pipe,shuffle=True,drop_last=True,batch_size=batch_size,collate_fn=None)



"""
on average 1kb can have 167 words with 5 letters and a space. so 5MB(around 250,000 words) should be enough for one vocab. 
"""

def create_src_tgt_vocab(dl, src_specials, tgt_specials,src_tokenizer,tgt_tokenizer, src_max_tokens=250000, tgt_max_tokens=250000,src_default='<unk>',tgt_default='<unk>'):
    src_counter = Counter()
    tgt_counter = Counter()

    for i, data in enumerate(dl):
        src_counter.update(src_tokenizer(data[0][0][0]))
        tgt_counter.update(tgt_tokenizer(data[0][1][0]))
        if i%100==0:
            print("progress....."+str(i))
    # First sort by descending frequency, then lexicographically
    src_sorted_by_freq_tuples = sorted(src_counter.items(), key=lambda x: (-x[1], x[0]))
    tgt_sorted_by_freq_tuples = sorted(tgt_counter.items(), key=lambda x: (-x[1], x[0]))

    """
    src  token processing
    """
    if src_max_tokens is None:
        src_ordered_dict = OrderedDict(src_sorted_by_freq_tuples)

    else:
        src_ordered_dict = OrderedDict(src_sorted_by_freq_tuples[: src_max_tokens - len(src_specials)])

    """
    tgt  token processing
    """
    if tgt_max_tokens is None:
        tgt_ordered_dict = OrderedDict(tgt_sorted_by_freq_tuples)

    else:
        tgt_ordered_dict = OrderedDict(tgt_sorted_by_freq_tuples[: tgt_max_tokens - len(tgt_specials)])

    src_vocab = vocab(src_ordered_dict,specials=src_specials)#, min_freq=src_min_freq, specials=src_specials, special_first=special_first)
    src_vocab.set_default_index(src_vocab[src_default])
    tgt_vocab = vocab(tgt_ordered_dict,specials=src_specials)#, min_freq=tgt_min_freq, specials=tgt_specials, special_first=special_first)
    tgt_vocab.set_default_index(tgt_vocab[tgt_default])
    return src_vocab, tgt_vocab

    """
    Save and retrieve the stats dict to save and load the vocab module again as it is a instance of nn.Module
    """

EN_VOCAB_PATH='../data/src_en_vocab.vocab'
FR_VOCAB_PATH='../data/tgt_fr_vocab.vocab'
seconds1=time.time()
data_path='R:\\studies\\transformers\\translation\\data\\en-fr.csv'
mode="train"
batch_size=1
tok_en=get_tokenizer('spacy', language='en_core_web_sm')
tok_fr=get_tokenizer('spacy', language='fr_core_news_sm')
# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

dp=get_data_pipes(data_path,mode=mode,batch_size=batch_size)
dl=get_data_loader(dp,batch_size=1)
src_vocab,tgt_vocab=create_src_tgt_vocab(dl,SPECIAL_SYMBOLS,SPECIAL_SYMBOLS,tok_en,tok_fr)

torch.save(src_vocab,EN_VOCAB_PATH)
torch.save(tgt_vocab,FR_VOCAB_PATH)


total_seconds=time.time()-seconds1
print("total time taken in seconds ... "+str(total_seconds))

