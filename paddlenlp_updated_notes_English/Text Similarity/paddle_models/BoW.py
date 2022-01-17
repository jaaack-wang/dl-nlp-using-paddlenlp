'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Simple BoW model for text matching using paddle. 
'''

import paddle 
import paddle.nn as nn


class BoW(nn.Layer):

    def __init__(self, 
                vocab_size, 
                output_dim,
                embedding_dim=100,
                padding_idx=0,  
                hidden_dim=50, 
                activation=nn.ReLU()):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.dense = nn.Linear(embedding_dim * 2, hidden_dim)
        self.activation = activation
        self.dense_out = nn.Linear(hidden_dim, output_dim)

    def encoder(self, embd):
        # summing up the embedding of ids to get text_embds
        # shape: ids_embd, (batch_size, max_text_len, embedding_dim)
        # text_embd, (batch_size, embedding_dim)
        return embd.sum(axis=1)

    def forward(self, text_a_ids, text_b_ids):
        # shape: text_ids, (batch_size, max_text_len) 
        # --> text_ids_embd, (batch_size, max_text_len, embedding_dim) 
        text_a_ids_embd = self.embedding(text_a_ids)
        text_b_ids_embd = self.embedding(text_b_ids)

        text_a_embd = self.encoder(text_a_ids_embd)
        text_b_embd = self.encoder(text_b_ids_embd)

        # concatenate [text_a_embd, text_b_embd]
        # shape: concat, (batch_size, embedding_dim * 2)
        concat = paddle.concat([text_a_embd, text_b_embd], axis=-1)

        # go through a dense layer before output
        # shape: hidden_out, (batch_size, hidden_dim)
        hidden_out = self.activation(self.dense(concat))

        # shape: out_logits, (batch_size, output_dim)
        # note that, since we will use cross entropy as the loss func, 
        # we will later use softmax to compute the loss
        out_logits = self.dense_out(hidden_out)
        return out_logits
