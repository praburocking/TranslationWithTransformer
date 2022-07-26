from torchdata.datapipes import iter
from torch.utils.data import DataLoader
import spacy
from collections import Counter,OrderedDict
from torchtext.vocab import vocab
from copy import deepcopy
import torch
from torchtext.data.utils import get_tokenizer
from buildVocab import FR_VOCAB_PATH,EN_VOCAB_PATH,SPECIAL_SYMBOLS
from torch.nn.utils.rnn import pad_sequence

def get_data_pipes(file_path: str, mode: str = "train", batch_size: int = 5000):
    buffer_size = 10000
    dp = iter.FileOpener([file_path], encoding='utf-8').parse_csv(skip_lines=1).shuffle(buffer_size=buffer_size)
    dp = dp.sharding_filter().batch(batch_size=batch_size, drop_last=True)
    dp=dp.map(convert_to_index)
    return dp

"""
collate_fn handles batch packing of the data
"""
class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch_data):
        #         breakpoint()
        source_data = [item[0] for item in batch_data]
        target_data = [item[1] for item in batch_data]
        source = pad_sequence(source_data, batch_first=True, padding_value=self.pad_idx)
        target = pad_sequence(target_data, batch_first=True, padding_value=self.pad_idx)
        return source, target

def get_data_loader(data_pipe,batch_size:int=5000,collate_fn=None):
   return DataLoader(data_pipe,shuffle=True,drop_last=True,batch_size=batch_size,collate_fn=collate_fn)


def convert_to_index(data):
    print(data)
    return [[[en_vocab(data[0][0])], [fr_vocab(data[0][1])]]]

data_path='R:\\studies\\transformers\\translation\\data\\en-fr.csv'
mode="train"
batch_size=1
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

dp=get_data_pipes(data_path,mode=mode,batch_size=batch_size)
dl=get_data_loader(dp,batch_size=1)


en_vocab=torch.load(EN_VOCAB_PATH)
fr_vocab=torch.load(FR_VOCAB_PATH)
