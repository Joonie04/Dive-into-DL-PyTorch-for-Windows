# 15.3 感情分析: 使用卷积神经网络
import os
import torch
from torch import nn
import collections
import numpy as np
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
    from downloader.aclImdb import get_dataset_path
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


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU则返回CPU
    
    返回:
        设备列表，包含所有可用的GPU或CPU
    """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


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


def predict_sentiment(net, vocab, sentence):
    """预测句子的情感
    
    参数:
        net: 训练好的模型
        vocab: 词表对象
        sentence: 待预测的句子
    
    返回:
        情感标签和概率
    """
    sentence = sentence.lower().replace(' ', '')
    tokens = tokenize([sentence], token='word')[0]
    indices = torch.tensor([vocab[tokens]])
    device = next(iter(net.parameters())).device
    indices = indices.to(device)
    with torch.no_grad():
        label = torch.argmax(net(indices), dim=1)
    return 'positive' if label.item() == 1 else 'negative'


batch_size = 64
train_iter, test_iter, vocab = load_data_imdb(batch_size)


## 15.3.1 一维卷积
def corr1d(X, K):
    """计算一维卷积
    
    参数:
        X: 输入张量，形状为 (n,)
        K: 卷积核，形状为 (k,)
    
    返回:
        输出张量，形状为 (n-k+1,)
    """
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])
print("corr1d(X, K): ", corr1d(X, K))


def corr1d_multi_in(X, K):
    """计算多个输入通道的一维卷积
    
    参数:
        X: 输入张量，形状为 (num_channels, n)
        K: 卷积核，形状为 (num_channels, k)
    
    返回:
        输出张量，形状为 (num_channels, n-k+1)
    """
    return torch.stack([corr1d(x, k) for x, k in zip(X, K)], dim=0)


X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
print("corr1d_multi_in(X, K): ", corr1d_multi_in(X, K))


## 15.3.3 textCNN模型
### 1. 定义模型
class TextCNN(nn.Module):
    """文本卷积神经网络模型
    
    使用不同大小的卷积核来捕获文本中的局部特征，然后通过最大池化和全连接层进行分类。
    
    参数:
        vocab_size: 词表大小
        embed_size: 词嵌入维度
        kernel_sizes: 卷积核大小列表
        num_channels: 每个卷积核的输出通道数列表
    """
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        
        self.dropout = nn.Dropout(0.5)
        
        self.decoder = nn.Linear(sum(num_channels), 2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.relu = nn.ReLU()
        
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        """前向传播
        
        参数:
            inputs: 输入张量，形状为 (batch_size, num_steps)
        
        返回:
            输出张量，形状为 (batch_size, 2)
        """
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        
        embeddings = embeddings.permute(0, 2, 1)
        
        encoding = torch.cat([self.relu(self.pool(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        
        outputs = self.decoder(self.dropout(encoding))
        return outputs


embed_size, kernel_sizes, num_channels = 100, [3, 4, 5], [100, 100, 100]
devices = try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, num_channels)


def init_weights(m):
    """初始化模型权重
    
    使用Xavier均匀初始化方法初始化线性层和卷积层的权重。
    
    参数:
        m: 模型模块
    """
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)


### 2. 加载预训练词向量
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


glove_embedding = TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
print("embeds.shape:", embeds.shape)

net.embedding.weight.data.copy_(torch.from_numpy(embeds))
net.constant_embedding.weight.data.copy_(torch.from_numpy(embeds))
net.constant_embedding.weight.requires_grad = False


### 3. 训练和评估模型
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

print("预测情感1:", predict_sentiment(net, vocab, 'this movie is so great'))
print("预测情感2:", predict_sentiment(net, vocab, 'this movie is so bad'))
