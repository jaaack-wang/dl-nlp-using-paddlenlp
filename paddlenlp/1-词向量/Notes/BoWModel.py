import paddle
# ---- couldn't find the source code for paddle.nn ----
import paddle.nn as nn
import paddlenlp


class BoWModel(nn.Layer):
    def __init__(self, embedder):
        super().__init__()

        # --------??? What does the embedder do????---------
        self.embedder = embedder
        emb_dim = self.embedder.embedding_dim
        self.encoder = paddlenlp.seq2vec.BoWEncoder(emb_dim)
        self.cos_sim_func = nn.CosineSimilarity(axis=-1)

    def get_cos_sim(self, text_a, text_b):
        text_a_embedding = self.forward(text_a)
        text_b_embedding = self.forward(text_b)

        # ---dot product of a & b divided by the prodct of their modulus---
        cos_sim = self.cos_sim_func(text_a_embedding, text_b_embedding)
        return cos_sim


    # ----text embedding = sum of the word embeddings----  
    def forward(self, text):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, embedding_dim)
        summed = self.encoder(embedded_text)

        return summed




# ============================= Usage ================================

# -------------------------------
# 1. load a token embedding model
# -------------------------------


from paddlenlp.embeddings import TokenEmbedding


# 初始化TokenEmbedding， 预训练embedding未下载时会自动下载并加载数据
token_embedding = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300")

# 查看token_embedding详情
print(token_embedding)


##### Notes: check embedding model that can be used
####
####from paddlenlp.embeddings import *
####print(list_embedding_name())
####
##### Or: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/embeddings/constant.py
####



# ------------------------------------------------
# 2. load the tk embedding model to BoWModel model
# ------------------------------------------------

model = BoWModel(embedder=token_embedding)



# ---------------------------------------------
# 3. Comparing cosine similarities of two texts
# ---------------------------------------------

# First: tokenizing a text, so a tokenizer is needed
# The Tokenizer can be found in another py. file in the same folder

from tokenizer import Tokenizer


tokenizer = Tokenizer()
# load the vocabulary
tokenizer.set_vocab(vocab=token_embedding.vocab)


# Second: convert the text to ids
# Suppose there are two texts named text_a and text_b
# ---???I don't understand why the text needs to be converted into ids???---
text_a_ids = paddle.to_tensor([tokenizer.text_to_ids(text_a)])
text_b_ids = paddle.to_tensor([tokenizer.text_to_ids(text_b)])


# Compare the cosine similaries of the two texts

print("text_a: {}".format(text_a))
print("text_b: {}".format(text_b))
print("cosine_sim: {}".format(model.get_cos_sim(text_a_ids, text_b_ids)










