# 15.4 自然语言推断与数据集
## 15.4.2 斯坦福自然语言推断（SNLI）数据集
import os
import re
import torch
from torch import nn
import collections


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
        return self._token_freqs

    @token_freqs.setter
    def token_freqs(self, value):
        self._token_freqs = value


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


def get_dataloader_workers():
    """获取数据加载器的工作进程数
    
    返回:
        工作进程数，通常为4
    """
    return 4


def download_extract(name):
    """获取数据集路径
    
    参数:
        name: 数据集名称
    
    返回:
        数据集目录路径
    """
    if name == 'SNLI':
        from downloader.snli import get_dataset_path
        return get_dataset_path()
    return None


### 1. 读取数据集
def read_snli(data_dir, is_train):
    """读取SNLI数据集
    
    SNLI（Stanford Natural Language Inference）数据集包含前提、假设和标签。
    标签有三种：entailment（蕴含）、contradiction（矛盾）、neutral（中立）。
    
    参数:
        data_dir: 数据集目录路径
        is_train: 是否读取训练集，True为训练集，False为测试集
    
    返回:
        (premises, hypotheses, labels) 元组
        - premises: 前提句子列表
        - hypotheses: 假设句子列表
        - labels: 标签列表，0表示蕴含，1表示矛盾，2表示中立
    """
    def extract_text(s):
        """提取文本
        
        清理文本中的括号和多余空格。
        
        参数:
            s: 原始文本
        
        返回:
            清理后的文本
        """
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt')
    
    with open(file_name, 'r', encoding='utf-8') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    
    return premises, hypotheses, labels


data_dir = download_extract('SNLI')
train_data = read_snli(data_dir, is_train=True)

print("训练数据示例:")
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('前提:', x0)
    print('假设:', x1)
    print('标签:', y)
    print()

test_data = read_snli(data_dir, is_train=False)

print("训练集标签分布:", [[row for row in train_data[2]].count(i) for i in range(3)])
print("测试集标签分布:", [[row for row in test_data[2]].count(i) for i in range(3)])


### 2. 定义用于加载数据集的类
class SNLIDataset(torch.utils.data.Dataset):
    """SNLI数据集类
    
    继承自PyTorch的Dataset类，用于加载和预处理SNLI数据集。
    
    参数:
        dataset: 原始数据集元组 (premises, hypotheses, labels)
        num_steps: 序列最大长度
        vocab: 词表对象，如果为None则创建新词表
    """
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        
        all_premise_tokens = tokenize(dataset[0], token='word')
        all_hypothesis_tokens = tokenize(dataset[1], token='word')
        
        if vocab is None:
            self.vocab = Vocab(all_premise_tokens + all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        
        print('read ' + str(len(self.labels)) + ' examples')

    def _pad(self, lines):
        """对词元序列进行截断或填充
        
        参数:
            lines: 词元列表的列表
        
        返回:
            截断或填充后的张量
        """
        return torch.tensor([truncate_pad(self.vocab[line], self.num_steps, self.vocab['<pad>']) for line in lines])
    
    def __getitem__(self, idx):
        """获取指定索引的数据样本
        
        参数:
            idx: 样本索引
        
        返回:
            (premise, hypothesis, label) 元组
        """
        return self.premises[idx], self.hypotheses[idx], self.labels[idx]
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.premises)


### 3. 整合代码
def load_data_snli(batch_size, num_steps=50):
    """加载SNLI数据集
    
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
    num_workers = get_dataloader_workers()
    data_dir = download_extract('SNLI')
    
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=num_workers)
    
    return train_iter, test_iter, train_set.vocab


train_iter, test_iter, vocab = load_data_snli(batch_size=128, num_steps=50)
print("词表大小:", len(vocab))

for X in train_iter:
    print("前提形状:", X[0].shape)
    print("假设形状:", X[1].shape)
    print("标签形状:", X[2].shape)
    break
