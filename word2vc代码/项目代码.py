# 创建时间    : 2026/3/12 16:14
# 作者       : 叶之瞳
# 文件名      : 项目代码.py
# 导入依赖
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np
import re
from collections import Counter
import time

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")


# 数据加载和预处理
def load_and_preprocess_data(file_path):
    """加载和预处理文本数据"""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取文本内容（去掉标签数字）
            content = re.sub(r'^\d+\s*', '', line.strip())
            if len(content) > 5:  # 只处理长度大于5的文本
                sentences.append(content)
    return sentences


# 加载数据
file_path = "Test.txt"  # 请确保文件路径正确
sentences = load_and_preprocess_data(file_path)
print(f"总共加载了 {len(sentences)} 条文本数据")
print("前5条数据示例：")
for i in range(min(5, len(sentences))):
    print(f"{i + 1}: {sentences[i][:50]}...")


# 中文分词处理
def chinese_tokenize(sentences):
    """对中文文本进行分词处理"""
    tokenized_sentences = []
    for i, sentence in enumerate(sentences):
        words = jieba.lcut(sentence)
        words = [word.strip() for word in words if len(word.strip()) > 1]
        tokenized_sentences.append(words)
        if i % 200 == 0:
            print(f"已处理 {i}/{len(sentences)} 条数据")
    return tokenized_sentences


tokenized_corpus = chinese_tokenize(sentences)
print("分词完成！")
print("\n分词后的数据示例：")
for i in range(min(3, len(tokenized_corpus))):
    print(f"原文: {sentences[i][:30]}...")
    print(f"分词: {tokenized_corpus[i][:10]}...")


# 词汇统计分析
def analyze_vocabulary(tokenized_corpus):
    """分析词汇统计信息"""
    all_words = [word for sentence in tokenized_corpus for word in sentence]
    word_freq = Counter(all_words)
    print("词汇统计信息：")
    print(f"总词汇量: {len(all_words)}")
    print(f"唯一词汇数: {len(word_freq)}")
    print(f"平均句子长度: {np.mean([len(sentence) for sentence in tokenized_corpus]):.2f}")
    print(f"最长句子长度: {max([len(sentence) for sentence in tokenized_corpus])}")
    print(f"最短句子长度: {min([len(sentence) for sentence in tokenized_corpus])}")
    print("\n前20个最高频词汇：")
    for word, freq in word_freq.most_common(20):
        print(f"{word}: {freq}次")
    return word_freq


word_frequency = analyze_vocabulary(tokenized_corpus)


# 构建词汇表
def build_vocab(tokenized_corpus, min_count=5):
    """构建词汇表和索引映射"""
    word_counts = Counter([word for sentence in tokenized_corpus for word in sentence])
    vocab = {word: count for word, count in word_counts.items() if count >= min_count}
    idx_to_word = ['<PAD>', '<UNK>'] + list(vocab.keys())
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    print(f"词汇表大小: {len(word_to_idx)} (包含 {len(word_counts) - len(vocab)} 个低频词被过滤)")
    return word_to_idx, idx_to_word, vocab


word_to_idx, idx_to_word, vocab = build_vocab(tokenized_corpus, min_count=5)


# 创建训练数据（负采样）
def create_training_data(tokenized_corpus, word_to_idx, window_size=5):
    """创建Word2Vec训练数据（Skip-gram with Negative Sampling）"""
    training_data = []
    vocab_size = len(word_to_idx)
    unk_idx = word_to_idx.get('<UNK>', 0)

    # 计算词频分布用于负采样
    word_counts = np.zeros(vocab_size)
    for word, idx in word_to_idx.items():
        if word in vocab:
            word_counts[idx] = vocab[word]

    # 负采样分布（按词频的3/4次方）
    word_distribution = np.power(word_counts, 0.75)
    word_distribution = word_distribution / word_distribution.sum()

    for sentence in tokenized_corpus:
        sentence_indices = [word_to_idx.get(word, unk_idx) for word in sentence]
        for i, target_word_idx in enumerate(sentence_indices):
            start = max(0, i - window_size)
            end = min(len(sentence_indices), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    context_word_idx = sentence_indices[j]
                    training_data.append((target_word_idx, context_word_idx))

    print(f"创建了 {len(training_data)} 个训练样本")
    return training_data, word_distribution


# 自定义Dataset
class Word2VecDataset(Dataset):
    def __init__(self, training_data, word_distribution, num_negatives=5):
        self.training_data = training_data
        self.word_distribution = word_distribution
        self.num_negatives = num_negatives
        self.vocab_size = len(word_distribution)

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        target, context = self.training_data[idx]
        negative_samples = []
        while len(negative_samples) < self.num_negatives:
            negative = np.random.choice(self.vocab_size, p=self.word_distribution)
            if negative != target and negative != context:
                negative_samples.append(negative)
        return {
            'target': torch.tensor(target, dtype=torch.long),
            'context': torch.tensor(context, dtype=torch.long),
            'negatives': torch.tensor(negative_samples, dtype=torch.long)
        }


# Word2Vec模型（Skip-gram with Negative Sampling）
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50):
        super(Word2VecModel, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        init_range = 0.5 / embedding_dim
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, target_word, context_word, negative_words):
        target_embed = self.target_embeddings(target_word)  # [batch, dim]
        context_embed = self.context_embeddings(context_word)  # [batch, dim]
        negative_embed = self.context_embeddings(negative_words)  # [batch, neg, dim]

        positive_score = torch.sum(target_embed * context_embed, dim=1)
        positive_score = torch.clamp(positive_score, max=10, min=-10)

        target_embed_expanded = target_embed.unsqueeze(1)  # [batch, 1, dim]
        negative_score = torch.bmm(negative_embed, target_embed_expanded.transpose(1, 2))
        negative_score = torch.clamp(negative_score.squeeze(2), max=10, min=-10)

        return positive_score, negative_score


# 损失函数
def skipgram_loss(positive_score, negative_score):
    positive_loss = -torch.log(torch.sigmoid(positive_score))
    negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_score)), dim=1)
    return (positive_loss + negative_loss).mean()
# 训练函数（修改后：接收参数字典并返回耗时和最终损失）
def train_word2vec_gpu(model, dataset, batch_size=1024, epochs=1, learning_rate=0.025, params=None):
    """在GPU上训练Word2Vec模型，返回模型、损失列表、训练耗时、最终损失"""
    model = model.to(device)
    if params:
        print("\n当前实验配置:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f"\n开始训练...")
    print(f"批量大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"优化器: Adam, 学习率: {learning_rate}")
    losses = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            target_words = batch['target'].to(device)
            context_words = batch['context'].to(device)
            negative_words = batch['negatives'].to(device)
            optimizer.zero_grad()
            positive_score, negative_score = model(target_words, context_words, negative_words)
            loss = skipgram_loss(positive_score, negative_score)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} 完成 | 平均损失: {avg_loss:.4f}")
    training_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {training_time:.2f}秒")
    final_loss = losses[-1] if losses else 0
    return model, losses, training_time, final_loss


# ========== 主程序：参数调优实验 ==========
# 定义一组实验参数（可根据需要修改）
experiment_params = {
    'vector_size': 200,  # 词向量维度
    'window': 8,  # 上下文窗口大小
    'min_count': 10,  # 最低词频过滤
    'batch_size': 1024,  # 批量大小
    'epochs': 5,  # 训练轮数
    'learning_rate': 0.01,  # 学习率
    'num_negatives': 10  # 负样本数量
}

# 创建数据集和模型
window = experiment_params['window']
min_count = experiment_params['min_count']
vector_size = experiment_params['vector_size']
num_negatives = experiment_params['num_negatives']
batch_size = experiment_params['batch_size']
epochs = experiment_params['epochs']
learning_rate = experiment_params['learning_rate']

# 注意：构建词汇表时使用的 min_count 应与实验参数一致
word_to_idx, idx_to_word, vocab = build_vocab(tokenized_corpus, min_count=min_count)

training_data, word_distribution = create_training_data(tokenized_corpus, word_to_idx, window_size=window)
dataset = Word2VecDataset(training_data, word_distribution, num_negatives=num_negatives)
model = Word2VecModel(vocab_size=len(word_to_idx), embedding_dim=vector_size)

# 训练
trained_model, losses, training_time, final_loss = train_word2vec_gpu(
    model, dataset,
    batch_size=batch_size,
    epochs=epochs,
    learning_rate=learning_rate,
    params=experiment_params
)

# 输出参数调优表格
print("\n" + "=" * 140)
print("参数调优记录")
print("=" * 140)
header = (f"{'实验批次':<8}{'vector_size':<12}{'window':<8}{'min_count':<10}"
          f"{'batch_size':<12}{'epochs':<8}{'learning_rate':<15}{'num_negatives':<13}"
          f"{'训练耗时(s)':<12}{'最终损失(困惑度替代)':<10}")
print(header)
print("-" * len(header))
# 假设这是第1次实验
print(f"{1:<8}{vector_size:<12}{window:<8}{min_count:<10}"
      f"{batch_size:<12}{epochs:<8}{learning_rate:<15.4f}{num_negatives:<13}"
      f"{training_time:<12.2f}{final_loss:<10.4f}")


# 提取词向量
def get_word_vectors(model, word_to_idx):
    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(len(word_to_idx)).to(device)
        word_vectors = model.target_embeddings(all_indices).detach().cpu()
    word_vectors_dict = {}
    for word, idx in word_to_idx.items():
        word_vectors_dict[word] = word_vectors[idx]
    return word_vectors_dict, word_vectors


word_vectors_dict, all_vectors = get_word_vectors(trained_model, word_to_idx)

# 保存词向量
torch.save(word_vectors_dict, "word_vectors.pt")
print("\n词向量已保存到 word_vectors.pt")

# 加载词向量（此处直接使用已生成的dict，无需重新加载）
loaded_word_vectors_dict = torch.load("word_vectors.pt")


# 创建与gensim兼容的Word2Vec包装器，并添加类比推理功能
class PyTorchWord2VecWrapper:
    def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
        self.wv = self.WordVectors(word_vectors_dict, word_to_idx, idx_to_word)
        self.vector_size = list(word_vectors_dict.values())[0].shape[0]

    class WordVectors:
        def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
            self.vectors_dict = word_vectors_dict
            self.word_to_idx = word_to_idx
            self.idx_to_word = idx_to_word
            self.key_to_index = word_to_idx
            self.vectors = np.stack([v.numpy() for v in word_vectors_dict.values()])

        def __getitem__(self, word):
            vec = self.vectors_dict.get(word, None)
            return vec.numpy() if vec is not None else None

        def __contains__(self, word):
            return word in self.vectors_dict

        def similarity(self, word1, word2):
            if word1 not in self.vectors_dict or word2 not in self.vectors_dict:
                raise KeyError(f"词语不在词汇表中: {word1} 或 {word2}")
            vec1 = self.vectors_dict[word1].numpy()
            vec2 = self.vectors_dict[word2].numpy()
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(vec1, vec2) / (norm1 * norm2)

        def most_similar(self, word, topn=10):
            if word not in self.vectors_dict:
                raise KeyError(f"词语不在词汇表中: {word}")
            target_vec = self.vectors_dict[word].numpy()
            similarities = []
            for w, vec_t in self.vectors_dict.items():
                if w == word:
                    continue
                vec = vec_t.numpy()
                norm_target = np.linalg.norm(target_vec)
                norm_vec = np.linalg.norm(vec)
                if norm_target == 0 or norm_vec == 0:
                    sim = 0.0
                else:
                    sim = np.dot(target_vec, vec) / (norm_target * norm_vec)
                similarities.append((w, sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:topn]

        def find_analogy(self, word_a, word_b, word_c, topn=10):
            """
            执行词类比推理：word_a 之于 word_b 相当于 word_c 之于 ?
            即寻找词d使得 vec(d) ≈ vec(word_b) - vec(word_a) + vec(word_c)
            """
            for w in [word_a, word_b, word_c]:
                if w not in self.vectors_dict:
                    raise KeyError(f"词语不在词汇表中: {w}")
            vec_a = self.vectors_dict[word_a].numpy()
            vec_b = self.vectors_dict[word_b].numpy()
            vec_c = self.vectors_dict[word_c].numpy()
            target_vec = vec_b - vec_a + vec_c
            # 计算与所有词的相似度，排除输入词
            similarities = []
            for w, vec_t in self.vectors_dict.items():
                if w in [word_a, word_b, word_c]:
                    continue
                vec = vec_t.numpy()
                norm_target = np.linalg.norm(target_vec)
                norm_vec = np.linalg.norm(vec)
                if norm_target == 0 or norm_vec == 0:
                    sim = 0.0
                else:
                    sim = np.dot(target_vec, vec) / (norm_target * norm_vec)
                similarities.append((w, sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:topn]


# 创建包装器
w2v_model = PyTorchWord2VecWrapper(loaded_word_vectors_dict, word_to_idx, idx_to_word)

print("\n" + "=" * 50)
print("词向量模型测试")
print("=" * 50)

# 相似词查找测试
test_words = ['中国', '美国', '基金', '房价']
print("\n相似词查找测试：")
for word in test_words:
    if word in w2v_model.wv:
        similar_words = w2v_model.wv.most_similar(word, topn=5)
        print(f"\n与'{word}'最相似的词：")
        for similar, score in similar_words:
            print(f"  {similar}: {score:.3f}")
    else:
        print(f"'{word}'不在词汇表中")

# 词汇相似度计算
print("\n词汇相似度计算：")
word_pairs = [('中国', '美国'), ('基金', '股票'), ('房价', '楼市'), ('北京', '上海')]
for word1, word2 in word_pairs:
    if word1 in w2v_model.wv and word2 in w2v_model.wv:
        similarity = w2v_model.wv.similarity(word1, word2)
        print(f"'{word1}' 和 '{word2}' 的相似度: {similarity:.3f}")
    else:
        print(f"词汇对 ({word1}, {word2}) 中有词不在词汇表中")

# ========== 类比推理测试 ==========
print("\n" + "=" * 50)
print("类比推理测试")
print("=" * 50)

analogy_tests = [
    ('中国', '北京', '美国'),  # 中国:北京 :: 美国:?
    ('男人', '女人', '国王'),  # 男人:女人 :: 国王:?
    ('医生', '医院', '教师'),  # 医生:医院 :: 教师:?
]

for a, b, c in analogy_tests:
    try:
        results = w2v_model.wv.find_analogy(a, b, c, topn=3)
        print(f"\n{a} : {b} 相当于 {c} : ?")
        for word, score in results:
            print(f"   → {word} (相似度: {score:.3f})")
    except KeyError as e:
        print(f"\n跳过类比 {a}:{b} :: {c}:? — {e}")

print("\n程序运行完毕！")