# dl-nlp-using-paddlenlp

[paddlenlp](https://github.com/PaddlePaddle/PaddleNLP)打卡笔记，待整理。所有的学习材料可在[基于深度学习的自然语言处理](https://aistudio.baidu.com/aistudio/course/introduce/24177)课程上找到（已结课），只需在[Paddle AI Studio](https://aistudio.baidu.com/aistudio/index)注册一个账号即可。[paddlenlp](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/tree/main/paddlenlp)文件夹内储存的是课程学习笔记和项目练习，但涉及到模型训练的部分，绝大大多数需要在GPU上运行，所以推荐直接在[Paddle AI Studio](https://aistudio.baidu.com/aistudio/index)平台上注册课程、直接运行。[paddlenlp_updated_notes_English](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/tree/main/paddlenlp_updated_notes_English/WordEmbedding)文件夹是我会陆续更新的新的、更为系统化的学习心得，并且也更容易看懂，可以直接在 [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb?utm_source=scs-index)或者你的电脑上运行。两个版本的笔记是不一样的。英语的笔记不是中文笔记的直接翻译。

<br>

Learning notes and materials on [paddlenlp](https://github.com/PaddlePaddle/PaddleNLP), taken during and after the [Deep Learning Based Natural Language Processing](https://aistudio.baidu.com/aistudio/course/introduce/24177) (The course is over, but you can still sign up for the course materials). The [paddlenlp](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/tree/main/paddlenlp) folder stores the course materials as well as my notes taken during the course (in Chinese), but most of the notebooks regarding model trainning need to be run on [Paddle AI Studio](https://aistudio.baidu.com/aistudio/index) platform for free GPU resources. The [paddlenlp_updated_notes_English](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/tree/main/paddlenlp_updated_notes_English/WordEmbedding) folder is a more systematic and beginner-friendly collection of notes that I will update regularly in English and can be run directly on [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb?utm_source=scs-index) or your own computers. Therefore, the contents of the two folders are very different. The English notes are not the translations to the Chinese ones. 


## 目录 Table of contents
(The links are for the updated English notes stored in [paddlenlp_updated_notes_English](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/tree/main/paddlenlp_updated_notes_English/WordEmbedding) folder that have been updated.)

<br>

- 1-词向量 [Word Embedding](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/tree/main/paddlenlp_updated_notes_English/WordEmbedding)
  - [loading pre-trained word embedding in paddlenp.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/WordEmbedding/1-loading%20pre-trained%20word%20embedding%20in%20paddlenp.ipynb)
  - [calculating text cosine similarity in paddlenlp.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/WordEmbedding/2-calculating%20text%20cosine%20similarity%20in%20paddlenlp.ipynb)
  - [embedding visualization using paddlepaddle tool.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/WordEmbedding/3-embedding%20visualization%20using%20paddlepaddle%20tool.ipynb)
  - [embedding visualization using tensorflow tool.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/WordEmbedding/3.2-embedding%20visualization%20using%20tensorflow%20tool.ipynb)
  - [training word embeddings using skip-gram with negative sampling in paddle.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/WordEmbedding/4-training%20word%20embeddings%20using%20skip-gram%20with%20negative%20sampling%20in%20paddle.ipynb)

<br>

- 2-文本语义相似度计算 [Text similarity](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/tree/main/paddlenlp_updated_notes_English/Text%20Similarity)
	- [Quick starts with paddle.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/Text%20Similarity/0%20-%20Quick_starts_with_paddle.ipynb)
	- [Get data.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/Text%20Similarity/1%20-%20get_data.ipynb)
	- [preprocess data.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/Text%20Similarity/2%20-%20preprocess_data.ipynb)
	- [wrapped up data preprocessor.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/Text%20Similarity/2.1%20-%20wrapped_up_data_preprocessor.ipynb)
	- [preprocess data using paddle.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/Text%20Similarity/2.2%20-%20preprocess_data_using_paddle.ipynb)

<br>

- 3-NER 命名实体识别 Named Entity Recognition

<br>

- 4-情感分析 [Sentiment analysis](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/tree/main/paddlenlp_updated_notes_English/Sentiment_Analysis)
	- [Quick starts with paddle.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/Sentiment_Analysis/0%20-%20very_quick_starts_with_paddle.ipynb)
	- [Get data.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/Sentiment_Analysis/1%20-%20get_data.ipynb)
	- [preprocess data.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/Sentiment_Analysis/2%20-%20preprocess_data.ipynb)
	- [wrapped up data preprocessor.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/Sentiment_Analysis/2.1%20-%20wrapped_up_data_preprocessor.ipynb)
	- [preprocess data using paddle.ipynb](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp/blob/main/paddlenlp_updated_notes_English/Sentiment_Analysis/2.2%20-%20preprocess_data_using_paddle.ipynb)

<br>

- 5-实体关系抽取 Relation extraction
- 6-问答系统/机器阅读理解/段落检索 Q&A system/Machine Reading Comprehension/passage retrieval 
- 7-结构化数据问答 Structured Q&A system
- 8-文本翻译 Machine Translation
- 9-机器同传 Machine Interpretation 
- 10-任务式对话 Task-Oriented Dialogue System
- 11-开放域对话 Open-domain Dialogue System
- 12-产业实践 Industry practice (pretraining)
