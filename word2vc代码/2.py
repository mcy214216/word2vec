import jieba
import re
from gensim.models import Word2Vec

# 读取文本并清洗
def read_and_clean(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', ' ', text)
    # 只保留中文字符和句子结束标点（用于切分）
    text = re.sub(r'[^\u4e00-\u9fff。！？]', '', text)
    # 按句子切分
    sentences = re.split(r'[。！？]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# 分词并生成训练数据（不进行停用词过滤）
def tokenize_sentences(sentences):
    tokenized = []
    for sent in sentences:
        words = jieba.lcut(sent)
        # 可选：过滤掉单字词（如“的”、“了”等），但不是必须的
        # words = [w for w in words if len(w) > 1]
        tokenized.append(words)
    return tokenized

file_path = 'sanguoyanyi.txt'
raw_sentences = read_and_clean(file_path)
sentences = tokenize_sentences(raw_sentences)

# 训练 Word2Vec（使用推荐的参数）
model = Word2Vec(sentences,
                 vector_size=200,   # 词向量维度
                 window=10,         # 上下文窗口大小
                 min_count=5,       # 忽略低频词
                 workers=4,         # 并行线程数
                 epochs=15,         # 迭代次数
                 # negative=10
                 )       # 负采样数
model.save('word2vec_model_improved.model')

# 测试
model = Word2Vec.load('word2vec_model_improved.model')
test_names = ['刘备', '曹操', '诸葛亮', '关羽']
for name in test_names:
    similar = model.wv.most_similar(name)
    print(f"\n与‘{name}’最相似的词：")
    for word, score in similar:
        print(f"{word}: {score:.4f}")

# 类比推理（可选）
analogy = model.wv.most_similar(positive=['刘备', '张飞'], negative=['关羽'], topn=5)
print("\n类比刘备+张飞-关羽：")
for word, score in analogy:
    print(f"{word}: {score:.4f}")