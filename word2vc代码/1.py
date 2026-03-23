# 创建时间    : 2026/3/9 23:49
# 作者       : 叶之瞳
# 文件名      : 1.py
import jieba
import re
import gensim
from gensim.models import Word2Vec

# 读取文本并清洗
def read_and_clean(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 只保留中文字符和句号（或其他分隔符），去除所有非中文字符
    text = re.sub(r'<[^>]*>', ' ', text)  # 把文本中<>标签替换为空格
    text = re.sub(r'[^\u4e00-\u9fff。，！？]', '', text)  # 可根据需要保留标点
    # 按句子切分（简单按句号分句）
    sentences = [s for s in text.split('。') if s.strip()]
    return sentences

# 分词并生成训练数据
def tokenize_sentences(sentences):
    tokenized = []
    for sent in sentences:
        words = jieba.lcut(sent)  # 使用 lcut 返回列表
        # 可选：去除停用词（需加载停用词表）
        # words = [w for w in words if w not in stopwords]
        tokenized.append(words)
    return tokenized

file_path = 'sanguoyanyi.txt'
raw_sentences = read_and_clean(file_path)
sentences = tokenize_sentences(raw_sentences)

# 训练 Word2Vec
model = Word2Vec(sentences, vector_size=200, window=8, min_count=5, workers=4, epochs=2
                 )
model.save('word2vec_model_improved.model')

# 测试
model = Word2Vec.load('word2vec_model_improved.model')
similar = model.wv.most_similar('刘备')
print("与‘刘备’最相似的词：")
for word, score in similar:
    print(f"{word}: {score:.4f}")

analogy_words=model.wv.most_similar(positive=['刘备','张飞'], negative=['关羽'], topn=10)
print(f'\n类⽐推理：刘备-张飞+关羽：')
for word, analogy in analogy_words:
    print(f"{word}: {analogy:.4f}")