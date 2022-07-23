from torchdata.datapipes import iter
from torch.utils.data import DataLoader
import spacy
from collections import Counter,OrderedDict
from torchtext.vocab import vocab
from copy import deepcopy
import torch
from torchtext.data.utils import get_tokenizer

def get_data_pipes(file_path: str, mode: str = "train", batch_size: int = 5000):
    buffer_size = 10000
    dp = iter.FileOpener([file_path], encoding='utf-8').parse_csv(skip_lines=1).shuffle(buffer_size=buffer_size)
    dp = dp.sharding_filter().batch(batch_size=batch_size, drop_last=True)
    # dp=dp.map(create_vocab)
    return dp




def get_data_loader(data_pipe,batch_size:int=5000):
   return DataLoader(data_pipe,shuffle=True,drop_last=True,batch_size=batch_size,collate_fn=None)


def create_vocab(data):
    print(data)
    # print(nlp_en(data[0][0]))
    # print(nlp_fr(data[1][0]))
    return [[[nlp_en(data[0][0])], [nlp_fr(data[0][1])]]]

"""
on average 1kb can have 167 words with 5 letters and a space. so 5MB(around 250,000 words) should be enough for one vocab. 
"""

def create_src_tgt_vocab(dl, src_specials, tgt_specials, src_max_tokens=250000, tgt_max_tokens=250000, src_min_freq=1,
                         tgt_min_freq=1, special_first=True,src_default='<UNK>',tgt_default='<UNK>'):
    src_counter = Counter()
    tgt_counter = Counter()

    for i, data in enumerate(dl):
        src_counter.update(nlp_en(data[0][0][0]))
        tgt_counter.update(nlp_fr(data[0][1][0]))
        if i==5:
            break

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
        src_ordered_dict = OrderedDict(tgt_sorted_by_freq_tuples[: tgt_max_tokens - len(tgt_specials)])

    src_vocab = vocab(src_ordered_dict)#,specials=src_specials)#, min_freq=src_min_freq, specials=src_specials, special_first=special_first)
    # src_vocab.set_default_index(src_vocab[src_default])
    tgt_vocab = vocab(tgt_ordered_dict)#,specials=src_specials)#, min_freq=tgt_min_freq, specials=tgt_specials, special_first=special_first)
    # tgt_vocab.set_default_index(tgt_vocab[tgt_default])
    return src_vocab, tgt_vocab

    """
    Save and retrieve the stats dict to save and load the vocab module again as it is a instance of nn.Module
    """

data_path='R:\\studies\\transformers\\translation\\data\\en-fr.csv'
mode="train"
batch_size=1
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")
# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

dp=get_data_pipes(data_path,mode=mode,batch_size=batch_size)
dl=get_data_loader(dp,batch_size=1)
src_vocab,tgt_vocab=create_src_tgt_vocab(dl, special_symbols,special_symbols)

# transformer.load_state_dict(torch.load("transformer_param"))
states_dict=deepcopy(src_vocab.state_dict())
torch.save(states_dict, "src_vocab_param")

states_dict=deepcopy(tgt_vocab.state_dict())
torch.save(states_dict, "tgt_vocab_param")

# from torchtext.vocab import vocab
# from collections import Counter, OrderedDict
# counter = Counter(["a", "a", "b", "b", "b"])
# sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
# ordered_dict = OrderedDict(sorted_by_freq_tuples)
# v1 = vocab(ordered_dict)
# print(v1['a']) #prints 1
# # print(v1['out of vocab']) #raise RuntimeError since default index is not set
# tokens = ['e', 'd', 'c', 'b', 'a']
# #adding <unk> token and default index
# unk_token = '<unk>'
# default_index = -1
# v2 = vocab(OrderedDict([(token, 1) for token in tokens]), specials=[unk_token])
# v2.set_default_index(default_index)
# print(v2['<unk>']) #prints 0
# print(v2['out of vocab']) #prints -1
# #make default index same as index of unk_token
# v2.set_default_index(v2[unk_token])
# v2['out of vocab'] is v2[unk_token] #prints True