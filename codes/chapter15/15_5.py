# 15.5 自然语言推断: 使用注意力
## 15.5.1 模型
import os
import re
import torch
from torch import nn
from torch.nn import functional as F
import collections
import numpy as np


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


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU则返回CPU
    
    返回:
        设备列表，包含所有可用的GPU或CPU
    """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def try_gpu():
    """返回GPU，如果没有GPU则返回CPU
    
    返回:
        设备对象
    """
    if torch.cuda.device_count() > 0:
        return torch.device('cuda')
    return torch.device('cpu')


def accuracy(y_hat, y):
    """计算预测正确的数量
    
    参数:
        y_hat: 预测结果，形状为 (batch_size, num_classes)
        y: 真实标签，形状为 (batch_size,)
    
    返回:
        预测正确的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的准确率
    
    参数:
        net: 神经网络模型
        data_iter: 数据迭代器
        device: 计算设备，如果为None则使用net的第一个设备
    
    返回:
        准确率和损失
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = [0.0, 0.0]
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = nn.CrossEntropyLoss()(y_hat, y)
            metric[0] += l * X.shape[0]
            metric[1] += accuracy(y_hat, y)
    return metric[0] / len(data_iter.dataset), metric[1] / len(data_iter.dataset)


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices):
    """使用多GPU训练模型
    
    参数:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        loss: 损失函数
        trainer: 优化器
        num_epochs: 训练轮数
        devices: 设备列表
    """
    timer = [0.0, 0.0]
    num_batches = len(train_iter)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    
    for epoch in range(num_epochs):
        metric = [0.0, 0.0, 0.0]
        for i, (features, labels) in enumerate(train_iter):
            timer[0] += 1
            trainer.zero_grad()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            l = loss(net(features), labels)
            l.sum().backward()
            trainer.step()
            with torch.no_grad():
                metric[0] += l * labels.shape[0]
                metric[1] += accuracy(net(features), labels)
                metric[2] += labels.shape[0]
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch {epoch + 1}, loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')


class TokenEmbedding:
    """词嵌入加载器
    
    用于加载预训练的词向量，如GloVe词向量。
    
    参数:
        embedding_name: 词向量名称，如 'glove.6b.100d'
    """
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
    
    def _load_embedding(self, embedding_name):
        """加载词向量文件
        
        参数:
            embedding_name: 词向量名称
        
        返回:
            (idx_to_token, idx_to_vec) 元组
        """
        idx_to_token = ['<unk>']
        idx_to_vec = [np.zeros(100)]
        
        try:
            glove_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'glove.6B.100d.txt')
            if os.path.exists(glove_path):
                with open(glove_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        token = parts[0]
                        vec = np.array([float(x) for x in parts[1:]], dtype='float32')
                        idx_to_token.append(token)
                        idx_to_vec.append(vec)
            else:
                print(f"警告: 未找到词向量文件 {glove_path}，将使用随机初始化")
        except Exception as e:
            print(f"警告: 加载词向量失败: {e}，将使用随机初始化")
        
        return idx_to_token, np.array(idx_to_vec)
    
    def __getitem__(self, tokens):
        """获取词元的词向量
        
        参数:
            tokens: 词元或词元列表
        
        返回:
            词向量或词向量列表
        """
        indices = [self.token_to_idx.get(token, 0) for token in tokens]
        vecs = self.idx_to_vec[indices]
        return vecs
    
    @property
    def token_to_idx(self):
        """词元到索引的映射"""
        return {token: idx for idx, token in enumerate(self.idx_to_token)}


### 1. 注意
def mlp(num_inputs, num_hiddens, flatten):
    """多层感知机（MLP）构建函数
    
    构建一个包含Dropout、线性层和ReLU激活函数的多层感知机。
    
    参数:
        num_inputs: 输入维度
        num_hiddens: 隐藏层维度
        flatten: 是否展平输出
    
    返回:
        MLP模型
    """
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)


class Attend(nn.Module):
    """注意力模块
    
    使用可分解注意力机制来计算前提和假设之间的注意力权重。
    通过MLP将输入映射到注意力空间，然后计算注意力权重。
    
    参数:
        num_inputs: 输入维度
        num_hiddens: 隐藏层维度
    """
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs=num_inputs, num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        """前向传播
        
        参数:
            A: 前提的嵌入表示，形状为 (batch_size, num_steps_A, embed_size)
            B: 假设的嵌入表示，形状为 (batch_size, num_steps_B, embed_size)
        
        返回:
            beta: 基于假设对前提的注意力加权，形状为 (batch_size, num_steps_A, embed_size)
            alpha: 基于前提对假设的注意力加权，形状为 (batch_size, num_steps_B, embed_size)
        """
        f_A = self.f(A)
        f_B = self.f(B)
        
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        
        return beta, alpha


### 2. 比较
class Compare(nn.Module):
    """比较模块
    
    将原始嵌入与注意力加权后的嵌入进行拼接，然后通过MLP进行比较。
    
    参数:
        num_inputs: 输入维度（通常是嵌入维度的2倍）
        num_hiddens: 隐藏层维度
    """
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs=num_inputs, num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        """前向传播
        
        参数:
            A: 前提的嵌入表示
            B: 假设的嵌入表示
            beta: 基于假设对前提的注意力加权
            alpha: 基于前提对假设的注意力加权
        
        返回:
            V_A: 前提的比较结果
            V_B: 假设的比较结果
        """
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B


### 3. 聚合
class Aggregate(nn.Module):
    """聚合模块
    
    将比较结果进行聚合，然后通过全连接层输出最终预测。
    
    参数:
        num_inputs: 输入维度
        num_hiddens: 隐藏层维度
        num_outputs: 输出维度（对于SNLI为3：entailment、contradiction、neutral）
    """
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs=num_inputs, num_hiddens=num_hiddens, flatten=False)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        """前向传播
        
        参数:
            V_A: 前提的比较结果
            V_B: 假设的比较结果
        
        返回:
            Y_hat: 预测结果，形状为 (batch_size, num_outputs)
        """
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat


### 4. 整个代码
class DecomposableAttention(nn.Module):
    """可分解注意力模型
    
    用于自然语言推断任务，包含三个主要部分：
    1. 注意（Attend）：计算前提和假设之间的注意力
    2. 比较（Compare）：比较原始嵌入和注意力加权后的嵌入
    3. 聚合（Aggregate）：聚合比较结果并输出预测
    
    参数:
        vocab: 词表对象
        embed_size: 词嵌入维度
        num_hiddens: 隐藏层维度
        num_inputs_attend: 注意模块的输入维度
        num_inputs_compare: 比较模块的输入维度
        num_inputs_agg: 聚合模块的输入维度
    """
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100, num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        """前向传播
        
        参数:
            X: 输入元组 (premises, hypotheses)
        
        返回:
            Y_hat: 预测结果
        """
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat


## 15.5.2 训练和评估模型
### 1. 读取数据集
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)

### 2. 创建模型
embed_size, num_hiddens, devices = 100, 200, try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)

glove_embedding = TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
print("embeds.shape:", embeds.shape)
net.embedding.weight.data.copy_(torch.from_numpy(embeds))

### 3. 训练和评估模型
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

### 4. 使用模型
def predict_snli(net, vocab, premise, hypothesis):
    """预测SNLI任务的标签
    
    参数:
        net: 训练好的模型
        vocab: 词表对象
        premise: 前提句子的词元列表
        hypothesis: 假设句子的词元列表
    
    返回:
        预测的标签：'entailment'、'contradiction' 或 'neutral'
    """
    net.eval()
    premise = torch.tensor(vocab[premise], device=try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)), hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 else 'neutral'


result = predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
print("预测结果:", result)
