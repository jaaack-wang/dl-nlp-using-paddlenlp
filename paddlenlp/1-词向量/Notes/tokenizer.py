import numpy as np
import jieba
import paddle

from collections import defaultdict
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab

class Tokenizer(object):
    def __init__(self):
        self.vocab = {}
        self.tokenizer = jieba
        self.vocab_path = 'vocab.txt'
        self.UNK_TOKEN = '[UNK]'
        self.PAD_TOKEN = '[PAD]'

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.tokenizer = JiebaTokenizer(vocab)

    def build_vocab(self, sentences):
        word_count = defaultdict(lambda: 0)
        for text in sentences:
            words = jieba.lcut(text)
            for word in words:
                word = word.strip()
                if word.strip() !='':
                    word_count[word] += 1

        word_id = 0
        for word, num in word_count.items():
            if num < 5:
                continue
            self.vocab[word] = word_id
            word_id += 1
        
        self.vocab[self.UNK_TOKEN] = word_id
        self.vocab[self.PAD_TOKEN] = word_id + 1
        self.vocab = Vocab.from_dict(self.vocab,
            unk_token=self.UNK_TOKEN, pad_token=self.PAD_TOKEN)
        # dump vocab to file
        self.dump_vocab(self.UNK_TOKEN, self.PAD_TOKEN)
        self.tokenizer = JiebaTokenizer(self.vocab)
        return self.vocab

    def dump_vocab(self, unk_token, pad_token):
        with open(self.vocab_path, "w", encoding="utf8") as f:
            for word in self.vocab._token_to_idx:
                f.write(word + "\n")
    
    def text_to_ids(self, text):
        input_ids = []
        unk_token_id = self.vocab[self.UNK_TOKEN]
        for token in self.tokenizer.cut(text):
            token_id = self.vocab.token_to_idx.get(token, unk_token_id)
            input_ids.append(token_id)

        return input_ids

    def convert_example(self, example, is_test=False):
        input_ids = self.text_to_ids(example['text'])

        if not is_test:
            label = np.array(example['label'], dtype="int64")
            return input_ids, label
        else:
            return input_ids

