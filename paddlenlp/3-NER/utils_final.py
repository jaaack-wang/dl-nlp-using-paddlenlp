import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad

def load_dict(dict_path):
    vocab = {}
    i = 0
    for line in open(dict_path, 'r', encoding='utf-8'):
        key = line.strip('\n')
        vocab[key] = i
        i+=1
    return vocab

def convert_example(example, tokenizer, label_vocab, max_seq_len=128):
    # # tokens, labels = example['tokens'], example['labels']
    # tokens, labels = example
    # tokenized_input = tokenizer(
    #     tokens, return_length=True, is_split_into_words=True, max_seq_len=max_seq_len)
    # # Token '[CLS]' and '[SEP]' will get label 'O'
    # labels = ['O'] + labels + ['O']
    # tokenized_input['labels'] = labels
    # # tokenized_input['labels'] = [label_vocab[x] for x in labels]
    # return tokenized_input['input_ids'], tokenized_input[
    #     'token_type_ids'], tokenized_input['seq_len'], tokenized_input['labels']
    labels = example['labels']
    tokens = example['tokens']
    no_entity_id = label_vocab['O']
    tokenized_input = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (
        len(tokenized_input['input_ids']) - len(tokenized_input['labels']))
    return tokenized_input['input_ids'], tokenized_input[
        'token_type_ids'], tokenized_input['seq_len'], tokenized_input['labels']


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval precision: %f - recall: %f - f1: %f" %
          (precision, recall, f1_score))
    model.train()

def predict(model, data_loader, ds, label_vocab):
    pred_list = []
    len_list = []
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(lens.numpy())
    preds = parse_decodes(ds, pred_list, len_list, label_vocab)
    return preds

# def parse_decodes(ds, decodes, lens, label_vocab):
    # decodes = [x for batch in decodes for x in batch]
    # lens = [x for batch in lens for x in batch]
    # id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    # outputs = []
    # for idx, end in enumerate(lens):
    #     sent = ds.data[idx][0][:end]
    #     tags = [id_label[x] for x in decodes[idx][1:end]]
    #     sent_out = []
    #     tags_out = []
    #     words = ""
    #     for s, t in zip(sent, tags):
    #         if t.endswith('-B') or t == 'O':
    #             if len(words):
    #                 sent_out.append(words)
    #             tags_out.append(t.split('-')[0])
    #             words = s
    #         else:
    #             words += s
    #     if len(sent_out) < len(tags_out):
    #         sent_out.append(words)
    #     outputs.append(''.join(
    #         [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    # return outputs
def parse_decodes(ds, decodes, lens, label_vocab):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lens):
        sent = ds.data[idx]['tokens'][:end]
        tags = [id_label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.startswith('B-') or t == 'O':
                if len(words):
                    sent_out.append(words)
                if t.startswith('B-'):
                    tags_out.append(t.split('-')[1])
                else:
                    tags_out.append(t)
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs
    
 
def predict(model, data_loader, ds, label_vocab):
    pred_list = []
    len_list = []
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(lens.numpy())
    preds = parse_decodes(ds, pred_list, len_list, label_vocab)
    return preds

