# 15.1 情感分析及数据集
import os
import torch
from torch import nn
import collections
import re
from matplotlib import pyplot as plt


def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元
    
    参数:
        lines: 文本行列表
        token: 词元类型，'word'或'char'
    
    返回:
        词元列表的列表
    """
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


def count_corpus(tokens):
    """统计词元频率
    
    参数:
        tokens: 词元列表
    
    返回:
        词元计数器
    """
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """词表类
    
    用于将词元转换为索引，以及将索引转换回词元。
    
    参数:
        tokens: 词元列表
        min_freq: 最小词频，低于此频率的词元将被过滤
        reserved_tokens: 保留词元列表，这些词元将始终包含在词表中
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self._unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """返回词表大小"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """获取词元对应的索引
        
        参数:
            tokens: 词元或词元列表
        
        返回:
            索引或索引列表
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self._unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """将索引转换为词元
        
        参数:
            indices: 索引或索引列表
        
        返回:
            词元或词元列表
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """返回未知词元索引"""
        return self._unk

    @property
    def token_freqs(self):
        """返回词元频率列表"""
        return self.token_freqs


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列
    
    参数:
        line: 词元索引列表
        num_steps: 目标序列长度
        padding_token: 填充词元索引
    
    返回:
        截断或填充后的序列
    """
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器
    
    参数:
        data_arrays: 数据数组元组
        batch_size: 批量大小
        is_train: 是否为训练数据，决定是否打乱数据
    
    返回:
        数据迭代器
    """
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小
    
    参数:
        figsize: 图表大小，默认为(3.5, 2.5)
    """
    plt.rcParams['figure.figsize'] = figsize


## 15.1.1 读取数据集
from downloader.aclImdb import get_dataset_path

data_dir = get_dataset_path()


def read_imdb(data_dir, is_train):
    """读取IMDB数据集文本序列和标签
    
    IMDB数据集包含电影评论，每个评论都被标记为正面（pos）或负面（neg）。
    
    参数:
        data_dir: 数据集目录路径
        is_train: 是否读取训练集，True为训练集，False为测试集
    
    返回:
        (data, labels) 元组
        - data: 评论文本列表
        - labels: 标签列表，1表示正面，0表示负面
    """
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


train_data = read_imdb(data_dir, is_train=True)
print("训练数据集:", len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print("标签:", y, "review:", x[0:60])


## 15.1.2 预处理数据集
train_tokens = tokenize(train_data[0], token='word')
vocab = Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])

set_figsize()
plt.xlabel('# tokens per review')
plt.ylabel('count')
plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50))
plt.show()

num_steps = 500
train_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print("train_features.shape", train_features.shape)


## 15.1.3 创建数据迭代器
train_iter = load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print("X:", X.shape, ", y:", y.shape)
    break
print("小批量数量:", len(train_iter))


## 15.1.4 整合代码
def load_data_imdb(batch_size, num_steps=500):
    """加载IMDB数据集
    
    整合数据下载、读取、预处理和迭代器创建的完整流程。
    
    参数:
        batch_size: 批量大小
        num_steps: 序列最大长度
    
    返回:
        (train_iter, test_iter, vocab) 元组
        - train_iter: 训练数据迭代器
        - test_iter: 测试数据迭代器
        - vocab: 词表对象
    """
    data_dir = get_dataset_path()
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = tokenize(train_data[0], token='word')
    test_tokens = tokenize(test_data[0], token='word')
    vocab = Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
    train_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = load_array((train_features, torch.tensor(train_data[1])), batch_size)
    test_iter = load_array((test_features, torch.tensor(test_data[1])), batch_size, is_train=False)
    return train_iter, test_iter, vocab


if __name__ == "__main__":
    print("=" * 60)
    print("15.1 情感分析及数据集")
    print("=" * 60)
    
    print("\n## 15.1.1 读取数据集")
    print(f"训练数据集: {len(train_data[0])} 条评论")
    for x, y in zip(train_data[0][:3], train_data[1][:3]):
        print(f"标签: {y}, review: {x[0:60]}")
    
    print("\n## 15.1.2 预处理数据集")
    print(f"词表大小: {len(vocab)}")
    print(f"训练特征形状: {train_features.shape}")
    
    print("\n## 15.1.3 创建数据迭代器")
    for X, y in train_iter:
        print(f"X: {X.shape}, y: {y.shape}")
        break
    print(f"小批量数量: {len(train_iter)}")
