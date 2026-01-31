# 15.2 情感分析: 使用循环神经网络
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch import nn
import time
import numpy as np
import matplotlib.pyplot as plt
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


def try_gpu(i=0):
    """尝试获取 GPU 设备
    
    参数:
        i: GPU 索引，默认为 0
    
    返回:
        GPU 设备，如果没有 GPU 则返回 CPU 设备
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """尝试获取所有可用的 GPU 设备
    
    返回:
        GPU 设备列表，如果没有 GPU 则返回包含 CPU 的列表
    """
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(torch.device(f'cuda:{i}'))
    return devices if devices else [torch.device('cpu')]


class Timer:
    """计时器类，用于记录和统计时间"""
    def __init__(self):
        """初始化计时器"""
        self.times = []
        self.start()
    
    def start(self):
        """开始计时"""
        self.tik = time.time()
    
    def stop(self):
        """停止计时并记录时间
        
        返回:
            本次计时的时长
        """
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        """计算平均时间
        
        返回:
            所有计时的平均值
        """
        return sum(self.times) / len(self.times)
    
    def sum(self):
        """计算总时间
        
        返回:
            所有计时的总和
        """
        return sum(self.times)
    
    def cumsum(self):
        """计算累积时间
        
        返回:
            所有计时的累积值列表
        """
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """累加器类，用于累加多个指标
    
    参数:
        n: 要累加的指标数量
    """
    def __init__(self, n):
        """初始化累加器
        
        参数:
            n: 要累加的指标数量
        """
        self.data = [0.0] * n
    
    def add(self, *args):
        """添加值到累加器
        
        参数:
            *args: 要添加的值
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        """重置累加器"""
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的值
        
        参数:
            idx: 索引
        
        返回:
            对应索引的值
        """
        return self.data[idx]


class Animator:
    """动画可视化类，用于绘制训练过程中的指标变化
    
    参数:
        xlabel: x 轴标签
        ylabel: y 轴标签
        legend: 图例
        xlim: x 轴范围
        ylim: y 轴范围
        xscale: x 轴缩放
        yscale: y 轴缩放
        fmts: 线条格式
        nrows: 子图行数
        ncols: 子图列数
        figsize: 图像大小
    """
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: self.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    
    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置坐标轴属性
        
        参数:
            ax: 坐标轴对象
            xlabel: x 轴标签
            ylabel: y 轴标签
            xlim: x 轴范围
            ylim: y 轴范围
            xscale: x 轴缩放
            yscale: y 轴缩放
            legend: 图例
        """
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()
    
    def add(self, x, y):
        """添加数据点
        
        参数:
            x: x 坐标
            y: y 坐标
        """
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.pause(0.001)
    
    @property
    def Y(self):
        """获取 Y 数据"""
        return self._Y if hasattr(self, '_Y') else None
    
    @Y.setter
    def Y(self, value):
        """设置 Y 数据"""
        self._Y = value


def accuracy(y_hat, y):
    """计算预测准确率
    
    参数:
        y_hat: 预测值
        y: 真实标签
    
    返回:
        准确率
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(torch.float32).sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的准确率
    
    参数:
        net: 神经网络模型
        data_iter: 数据迭代器
        device: 设备
    
    返回:
        准确率
    """
    if isinstance(net, nn.Module):
        net.eval()
    if not device:
        device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch13(net, train_iter, test_iter, loss_fn, trainer, num_epochs, devices):
    """训练模型
    
    参数:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        loss_fn: 损失函数
        trainer: 优化器
        num_epochs: 训练轮数
        devices: 设备列表
    """
    timer, num_batches = Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = None, None
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            outputs = net(features)
            l = loss_fn(outputs, labels)
            l.sum().backward()
            trainer.step()
            with torch.no_grad():
                acc = accuracy(outputs, labels)
            metric.add(l.sum(), l.numel(), acc, labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[1], metric[2] / metric[3], None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[1]:.3f}, train acc {metric[2] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')


class TokenEmbedding:
    """词元嵌入类
    
    用于加载和管理预训练的词向量，如 GloVe 词向量。
    提供了词元到索引、索引到词元的映射，以及获取词向量的功能。
    
    参数:
        embedding_name: 嵌入名称，用于指定要加载的词向量数据集
    """
    def __init__(self, embedding_name):
        """初始化词元嵌入
        
        参数:
            embedding_name: 嵌入名称
        """
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        """加载词嵌入
        
        从本地数据集目录加载预训练的词向量文件。
        词向量文件格式为：每行一个词，第一个字段是词，后面是词向量值。
        
        参数:
            embedding_name: 嵌入名称
        
        返回:
            (idx_to_token, idx_to_vec) 元组
            - idx_to_token: 索引到词元的列表
            - idx_to_vec: 索引到词向量的张量
        """
        from pathlib import Path
        idx_to_token, idx_to_vec = ['<unk>'], []
        
        if embedding_name == 'glove.6b.50d':
            from downloader.glove_6b_50d import get_dataset_path
            data_dir = get_dataset_path()
            vec_file = Path(data_dir) / "glove.6B.50d.txt"
        elif embedding_name == 'glove.6b.100d':
            from downloader.glove_6b_100d import get_dataset_path
            data_dir = get_dataset_path()
            vec_file = Path(data_dir) / "glove.6B.100d.txt"
        elif embedding_name == 'glove.42b.300d':
            from downloader.glove_42b_300d import get_dataset_path
            data_dir = get_dataset_path()
            vec_file = Path(data_dir) / "glove.42B.300d.txt"
        elif embedding_name == 'wiki.en':
            from downloader.wiki_en import get_dataset_path
            data_dir = get_dataset_path()
            vec_file = Path(data_dir) / "wiki.en.vec"
        else:
            raise ValueError(f"未知的嵌入名称: {embedding_name}")

        with open(vec_file, 'r', encoding='utf-8') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        """获取词元的词向量
        
        参数:
            tokens: 单个词元或词元列表
        
        返回:
            词向量张量
        """
        indices = [self.token_to_idx.get(token, self.unknown_idx) for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        """返回词表大小
        
        返回:
            词表中的词元数量
        """
        return len(self.idx_to_token)


batch_size = 64
train_iter, test_iter, vocab = load_data_imdb(batch_size)


## 15.2.1 使用循环神经网络表示单个文本
class BiRNN(nn.Module):
    """双向循环神经网络模型
    
    使用双向 LSTM 对文本进行编码，然后通过全连接层进行情感分类。
    
    参数:
        vocab_size: 词表大小
        embed_size: 词嵌入维度
        num_hiddens: 隐藏层维度
        num_layers: LSTM 层数
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        """前向传播
        
        参数:
            inputs: 输入词元索引张量，形状为 (num_steps, batch_size)
        
        返回:
            输出张量，形状为 (batch_size, 2)
        """
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


embed_size, num_hiddens, num_layers = 100, 100, 2
device = try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)


def init_weights(m):
    """初始化模型权重
    
    使用 Xavier 均匀分布初始化线性层和 LSTM 层的权重。
    
    参数:
        m: 模型模块
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


net.apply(init_weights)


## 15.2.2 加载预训练的词向量
glove_embedding = TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
print("embeds.shape:", embeds.shape)

net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False


## 15.2.3 训练和评估模型
def predict_sentiment(net, vocab, sequence):
    """预测文本序列的情感
    
    参数:
        net: 训练好的模型
        vocab: 词表对象
        sequence: 输入文本序列
    
    返回:
        情感标签，'positive' 或 'negative'
    """
    sequence = torch.tensor(vocab[sequence.split()], device=try_gpu())
    output = net(sequence.reshape(1, -1))
    label = torch.argmax(output, dim=1)
    return 'positive' if label.item() == 1 else 'negative'


if __name__ == "__main__":
    print("=" * 60)
    print("15.2 情感分析: 使用循环神经网络")
    print("=" * 60)
    
    print("\n## 15.2.1 使用循环神经网络表示单个文本")
    print(f"词表大小: {len(vocab)}")
    print(f"模型参数: embed_size={embed_size}, num_hiddens={num_hiddens}, num_layers={num_layers}")
    
    print("\n## 15.2.2 加载预训练的词向量")
    print(f"GloVe 词向量形状: {embeds.shape}")
    
    print("\n## 15.2.3 训练和评估模型")
    print("训练开始...")
    
    lr, num_epochs = 0.01, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, device)
    
    print("\n预测示例:")
    print(f"  'this movie is so great' -> {predict_sentiment(net, vocab, 'this movie is so great')}")
    print(f"  'this movie is so bad' -> {predict_sentiment(net, vocab, 'this movie is so bad')}")
