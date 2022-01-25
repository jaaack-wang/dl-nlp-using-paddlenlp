'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Utility functions for my repository: text-classification-explained
'''
from collections import Counter, defaultdict
from collections.abc import Iterable
import json
from os.path import join
from os.path import exists
import re
import numpy as np


def load_dataset(fpath, num_row_to_skip=1):
    def read(path):
        data = open(path)
        for _ in range(num_row_to_skip):
            next(data)
    
        for line in data:
            line = line.split('\t')
        
            yield line[1].rstrip(), int(line[0])
    
    if isinstance(fpath, str):
        assert exists(fpath), f"{fpath} does not exist!"
        return list(read(fpath))
    
    elif isinstance(fpath, (list, tuple)):
        for fp in fpath:
            assert exists(fp), f"{fp} does not exist!"
        return [list(read(fp)) for fp in fpath]
    
    raise TypeError("Input fpath must be a (list) of valid filepath(es)")


def gather_text(dataset, end_col=1):
    out = []
    for data in dataset:
        out.extend(data[:end_col])
    return out


class TextVectorizer:
    
    def __init__(self, tokenizer, preprocessor=None):
        if preprocessor:
            self.tokenize = lambda tx: tokenizer(preprocessor(tx))
        else:
            self.tokenize = tokenizer        
        
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self.vocab_freq_count = None
    
    def _create_vocab_dicts(self, unique_tks):
        for tk in ['[PAD]', '[UNK]']:
            if tk in unique_tks:
                unique_tks.remove(tk)
        
        unique_tks = ['[PAD]', '[UNK]'] + unique_tks
        
        self.vocab_to_idx = {tk: i for i, tk in enumerate(unique_tks)}
        self.vocab_to_idx = defaultdict(lambda: 1, self.vocab_to_idx)
        
        self.idx_to_vocab = {i: v for v, i in self.vocab_to_idx.items()}
        
        print('Two vocabulary dictionaries have been built!\n' \
             + 'Please call \033[1mX.vocab_to_idx | X.idx_to_vocab\033[0m to find out more' \
             + ' where [X] stands for the name you used for this TextVectorizer class.')
        
    
    def build_vocab(self, text, top=None, random=False):
        
        if isinstance(text, str):
            tks = self.tokenize(text)

        elif isinstance(text, (list, tuple,)):
            assert all(isinstance(t, str) for t in text), f'text must be a list/tuple of str'
        
            # alternatively, sum([], [tokenize(t) for t in text]) does the job, 
            # but it is much slower when the list is a large one 
            tks = []
            for t in text:
                tks.extend(self.tokenize(t))
        else:  
            raise TypeError(f'Input text must be str/list/tuple, but {type(text)} was given')
    
        if random:
            make_vocab = lambda tks: list(set(tks))[:top]
        else:
            self.vocab_freq_count = Counter(tks).most_common()
            make_vocab = lambda tks: [tk[0] for tk in self.vocab_freq_count][:top]
        
        unique_tks =  make_vocab(tks)
        
        self._create_vocab_dicts(unique_tks)
    
    def text_encoder(self, text, vectorized=False, dtype="int64"):
        if isinstance(text, list):
            return [self(t, vectorized, dtype) for t in text]
        
        tks = self.tokenize(text)
        out = [self.vocab_to_idx[tk] for tk in tks]
        if vectorized:
            out = np.asarray(out, dtype=dtype)
            
        return out
    
    def text_decoder(self, text_ids, sep=' '):
        if all(isinstance(ids, Iterable) for ids in text_ids):
            return [self.text_decoder(ids, sep) for ids in text_ids]
            
        out = []
        for text_id in text_ids:
            out.append(self.idx_to_vocab[text_id])
            
        return f'{sep}'.join(out)
    
    def _save_json_file(self, dic, fpath):
        if exists(fpath):
            print(f"{fpath} already exists. Do you want to overwrite it?")
            print("Press [N/n] for NO, or [any other key] to overwrite.")
            confirm = input()
            if confirm in ['N', 'n']:
                return
        
        with open(fpath, 'w') as f:
            json.dump(dic, f)
            print(f"{fpath} has been successfully saved!")    
    
    def save_vocab_as_json(self, 
                           v_to_i_fname='vocab_to_idx.json', 
                           i_to_v_fname='idx_to_vocab.json'):
        
        fmt_conv = lambda x: x + '.json' if not x.endswith('.json') else x
        
        v_to_i_fname, i_to_v_fname = fmt_conv(v_to_i_fname), fmt_conv(i_to_v_fname)
        
        self._save_json_file(self.vocab_to_idx, v_to_i_fname)
        self._save_json_file(self.idx_to_vocab, i_to_v_fname)
        
    def load_vocab_from_json(self, 
                             v_to_i_fname='vocab_to_idx.json', 
                             i_to_v_fname='idx_to_vocab.json'):
        
        fp1, fp2 = v_to_i_fname, i_to_v_fname
        if exists(fp1):
            vocab_to_idx = json.load(open(fp1))
            self.vocab_to_idx = defaultdict(lambda: 1, vocab_to_idx)
            print(f"{fp1} has been successfully loaded!" \
                 + " Please call \033[1mX.vocab_to_idx\033[0m to find out more.")
        
        if exists(fp2):
            self.idx_to_vocab = json.load(open(fp2))
            self.idx_to_vocab = {int(idx): tk for idx, tk in self.idx_to_vocab.items()}
            print(f"{fp2} has been successfully loaded!" \
                 + " Please call \033[1mX.idx_to_vocab\033[0m to find out more.")
            
        if not self.vocab_to_idx and not self.idx_to_vocab:
            raise RuntimeError(f"both {fp1} and {fp2} do not exist. Please double check the filename!")
        
        elif not self.vocab_to_idx:
            self.vocab_to_idx = defaultdict(lambda: 1, {tk: idx for idx, tk in self.idx_to_vocab.items()})
            print(f"{fp1} does not exist, but has been been successfully built from X.idx_to_vocab." \
                 + " Please call \033[1mX.vocab_to_idx\033[0m to find out more.")
        elif not self.idx_to_vocab:
            self.idx_to_vocab = {idx: tk for tk, idx in self.vocab_to_idx.items()}
            print(f"{fp2} does not exist, but has been been successfully built from X.vocab_to_idx." \
                 + " Please call \033[1mX.idx_to_vocab\033[0m to find out more.")
            
        print('\nWhere [X] stands for the name you used for this TextVectorizer class.')
        
    
    def build_vocab_from_list_tks(self, lst):
        
        assert isinstance(lst, list), f"Input lst must be list, not {type(lst)}"
        
        self.vocab_freq_count = None
        unique_tks = list(set(lst))
        
        self._create_vocab_dicts(unique_tks)
        
    def __call__(self, text, vectorized=False, dtype="int64"):
        if not self.vocab_to_idx:
            try:
                self.load_vocab_from_json()
            except:
                raise ValueError("No vocab is built or loaded.")
            
        return self.text_encoder(text)


def encode_dataset(dataset, encoder):
    
    for text, label in dataset:
        text = encoder(text)
        
        yield [text, label]


def batch_creater(lst, 
                  batch_size, 
                  reminder_threshold=0):
    
    assert batch_size > 1, "batch_size must be greater than 1"
    
    lst_size = len(lst)
    reminder = lst_size % batch_size
    if reminder_threshold > reminder:
        lst = lst[:-reminder]
        lst_size, reminder = len(lst), 0
    
    batch_num = lst_size // batch_size
    end_idx = batch_num + 1 if reminder else batch_num
    
    out = []
    for i in range(0, end_idx):
        out.append(lst[i * batch_size : (i+1) * batch_size])

    if reminder == 1:
        out[-1] = list(out[-1])
    
    return out


def build_batches(dataset, 
                  batch_size, 
                  shuffle=True, 
                  pad_idx=0, 
                  max_seq_len=None, 
                  dtype="int64", 
                  reminder_threshold=0, 
                  include_seq_len=False):
    
    # ------------- building bacthes first -------------
    
    if shuffle:
        np.random.shuffle(dataset)
        
    batches = batch_creater(dataset, batch_size, reminder_threshold)

    
    # ------------- start padding -------------
    # we can reuse the pad func above but the following is more efficient
    
    def _pad(lst, max_len):
        dif = max_len - len(lst)
        if dif > 0:
            return lst + [pad_idx] * dif
        if dif < 0:
            return lst[:max_len]
        return lst
        
    
    def pad(batch):
        if max_seq_len:
            max_len = max_seq_len
        else:
            max_len = max(len(b[0]) for b in batch)
            
        text, label = [], []
        
        if include_seq_len:
            if max_seq_len:
                text_len = [len(bt[0]) if len(bt[0]) < max_seq_len 
                              else max_seq_len for bt in batch]
            else:
                text_len = [len(bt[0]) for bt in batch]
                
            text_len = np.asarray(text_len, dtype=dtype)            
        
        for idx, bt in enumerate(batch):
            
            # ----- for text a -----
            text.append(_pad(bt[0], max_len))

            # ----- for the label -----
            label.append(bt[-1])
        
        text = np.asarray(text, dtype=dtype)
        label = np.asarray(label, dtype=dtype)
        
        if include_seq_len:
            return [text, text_len, label]
        
        return [text, label]
    
    out = []
    for batch in batches:
        out.append(pad(batch))
    
    return out
