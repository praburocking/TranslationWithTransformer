import numpy as np
import pandas as pd


def read_file_to_list(path):
    with open(path, encoding="utf8") as f:
        content_list = f.readlines()
    content_list = [x.strip() for x in content_list]
    return content_list


def get_data(src_file_path, tgt_file_path, src_lang_name, tgt_lang_name):
    raw_src = read_file_to_list(src_file_path)
    raw_tgt = read_file_to_list(tgt_file_path)
    data = pd.DataFrame({src_lang_name: raw_src, tgt_lang_name: raw_tgt})

    return data

def write_data(path,data):
    with open(path, 'w', encoding="utf8") as fp:
        for line in data:
            # write each item on a new line
            fp.write("%s\n" % line)



def split_data(data,val_frac=0.1):
    val_split_idx = int(len(data) * val_frac)  # index on which to split
    data_idx = list(range(len(data)))  # create a list of ints till len of data
    np.random.shuffle(data_idx)
    val_idx = data_idx[:val_split_idx]
    train_idx = data_idx[val_split_idx:]
    train_data = data.iloc[train_idx].reset_index().drop('index', axis=1)
    val_data = data.iloc[val_idx].reset_index().drop('index', axis=1)
    return train_data, val_data



def train_val_split():
    eng_file_path = '../data/eng_trans.txt'
    fr_file_path = '../data/french_trans.txt'
    src_lang_name = 'en'
    tgt_lang_name = 'fr'
    src_lang_vocab_path='./vocab/src_vocab.pkl'
    tgt_lang_vocab_path='./vocab/tgt_vocab.pkl'

    data = get_data(eng_file_path, fr_file_path, src_lang_name, tgt_lang_name)
    train_data,val_data=split_data(data)
    write_data('../data/train_eng_trans.txt',train_data['en'])
    write_data('../data/train_french_trans.txt',train_data['fr'])
    write_data('../data/val_eng_trans.txt',val_data['en'])
    write_data('../data/val_french_trans.txt',val_data['fr'])



