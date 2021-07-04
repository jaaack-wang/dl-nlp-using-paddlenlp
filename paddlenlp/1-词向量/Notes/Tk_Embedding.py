from paddlenlp.embeddings import TokenEmbedding


# -----------------------------------------------
# 1. Initialization: load a token embedding model
# -----------------------------------------------

# default: w2v.baidu_encyclopedia.target.word-word.dim300
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



# -------
# 2.Usage
# -------


# 2.1: Search the embedding of a given word

test_token_embedding = token_embedding.search("中国")
print(test_token_embedding)


# 2.1: calculate the cosine similarity of given words

score1 = token_embedding.cosine_sim("女孩", "女人")
score2 = token_embedding.cosine_sim("女孩", "书籍")
print('score1:', score1)
print('score2:', score2)






