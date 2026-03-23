# 创建时间    : 2026/3/9 08:32
# 作者       : 叶之瞳
# 文件名      : word2vc代码.py
# from  gensim.models import KeyedVectors
#
# # 加载模型
# pretranined_model_path='GoogleNews-vectors-negative300.bin.gz'
# model = KeyedVectors.load_word2vec_format(pretranined_model_path, binary=True)
#
# word1 = 'cat'
# word2 = 'dog'
# similarity = model.similarity(word1, word2)
# print(f"{word1} 和 {word2} 的相似度为：{similarity:.4f}")
#
# worda='king'
# wordb='man'
# wordc='woman'
# result = model.most_similar(positive=[wordb, wordc], negative=[worda])
# print(f"The result for the analogy '{worda}:{wordb}::{wordc}:' is {result[0][0]}")

#2使用三国演义文本来训练word2vec模型
import jieba
import re

#定义一个函数来读取中文文本文件并转换为句子列表
def read_chinese_file_to_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        text = re.sub(r'<[^>]*>', ' ', text)#把文本中<>标签替换为空格
        text = re.sub(r'[^\s\d\w\u4e00-\u9fff]', '', text)#把文本中非空格、非数字、非单词、非中文字符替换为空
        text = re.sub(r'\s+', ' ', text)#把文本中多个空格替换为一个空格
        text =re.sub(r'\n', '', text)#把文本中换行符替换为空


        words = jieba.cut(text)
        sentences=[list(words)]#将分词结果转换为句子列表
    return sentences

import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import Word2Vec

#文件路径
file_path = 'sanguoyanyi.txt'
#调用函数句子并获取句子列表
sentences = read_chinese_file_to_sentences(file_path)
#创建并训练Word2Vec模型
model = Word2Vec(sentences, vector_size=1000, window=5, min_count=8, workers=4,epochs=10)
#获取单词的向量表示
word_vector = model.wv
# 保存模型
model.save('word2vec_model.model')

# 读取模型
model = Word2Vec.load('word2vec_model.model')
#找出与某个单词最相似的词
a='刘备'
similar_words = model.wv.most_similar(a)
print("语义相似性：")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")

#类⽐推理
analogy_words=model.wv.most_similar(positive=['刘备','张飞'], negative=['关羽'], topn=10)
print(f'\n类⽐推理：刘备-张飞+关羽：')
for word, analogy in analogy_words:
    print(f"{word}: {analogy:.4f}")
