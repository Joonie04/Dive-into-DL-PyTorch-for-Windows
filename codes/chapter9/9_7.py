# 9.7 序列到序列学习（Seq2Seq）

import os  # 导入os模块，用于文件路径操作
import sys  # 导入sys模块，用于系统路径操作
import collections  # 导入collections模块，用于计数和默认字典
import math  # 导入math模块，用于数学计算
import time  # 导入time模块，用于计时
import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
from torch.utils import data  # 导入PyTorch数据工具
from matplotlib import pyplot as plt  # 导入matplotlib.pyplot绘图模块

# 添加downloader目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'downloader'))  # 将downloader目录添加到Python路径

from fra_eng import download  # 导入下载函数

# ========== 从9_5.py复制的数据处理方法 ==========

# 定义读取"英语-法语"数据集的函数
def read_data_nmt():  # 定义读取"英语-法语"数据集的函数
    """载入"英语-法语"数据集"""
    data_dir = download('fra-eng')  # 下载并解压"fra-eng"数据集
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:  # 以只读模式打开fra.txt文件，使用UTF-8编码
        return f.read()  # 读取并返回文件内容

# 定义预处理"英语-法语"数据集的函数
def preprocess_nmt(text):  # 定义预处理"英语-法语"数据集的函数
    """预处理"英语-法语"数据集"""
    def no_space(char, prev_char):  # 定义判断是否需要空格的内部函数
        return char in set(',.!?') and prev_char != ' '  # 如果字符是标点符号且前一个字符不是空格，则不需要空格

    # 使用空格替换不间断空格（\u202f是不间断空格，\xa0是不换行空格）
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()  # 替换特殊空格并转换为小写
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]  # 在标点符号前插入空格
    return ''.join(out)  # 将字符列表拼接成字符串

# 定义词元化"英语-法语"数据集的函数
def tokenize_nmt(text, num_examples=None):  # 定义词元化"英语-法语"数据集的函数
    """词元化"英语-法语"数据集"""
    source, target = [], []  # 初始化源语言（英语）和目标语言（法语）的列表
    for i, line in enumerate(text.split('\n')):  # 遍历每一行
        if num_examples and i > num_examples:  # 如果指定了示例数量且已超过
            break  # 跳出循环
        parts = line.split('\t')  # 按制表符分割行（英语和法语之间用制表符分隔）
        if len(parts) == 2:  # 如果分割后有两部分
            source.append(parts[0].split(' '))  # 将英语部分按空格分割并添加到source列表
            target.append(parts[1].split(' '))  # 将法语部分按空格分割并添加到target列表
    return source, target  # 返回源语言和目标语言的词元列表

# 定义词表类
class Vocab:  # 定义词表类
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):  # 初始化函数
        if tokens is None:  # 如果没有提供词元
            tokens = []  # 初始化为空列表
        if reserved_tokens is None:  # 如果没有提供保留词元
            reserved_tokens = []  # 初始化为空列表
        counter = count_corpus(tokens)  # 统计词元频率
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)  # 按频率降序排序词元
        self._unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens  # 初始化未知词元索引和唯一词元列表
        uniq_tokens += [token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]  # 添加频率大于等于min_freq的词元
        self.idx_to_token, self.token_to_idx = [], dict()  # 初始化索引到词元和词元到索引的映射
        for token in uniq_tokens:  # 遍历唯一词元
            self.idx_to_token.append(token)  # 添加词元到索引列表
            self.token_to_idx[token] = len(self.idx_to_token) - 1  # 添加词元到索引映射

    def __len__(self):  # 定义长度函数
        return len(self.idx_to_token)  # 返回词表大小

    def __getitem__(self, tokens):  # 定义索引函数
        if not isinstance(tokens, (list, tuple)):  # 如果tokens不是列表或元组
            return self.token_to_idx.get(tokens, self._unk)  # 返回词元的索引，如果不存在则返回未知词元索引
        return [self.__getitem__(token) for token in tokens]  # 返回词元列表的索引列表

    def to_tokens(self, indices):  # 定义索引到词元的函数
        if not isinstance(indices, (list, tuple)):  # 如果indices不是列表或元组
            return self.idx_to_token[indices]  # 返回索引对应的词元
        return [self.idx_to_token[index] for index in indices]  # 返回索引列表对应的词元列表

    @property  # 定义属性装饰器
    def unk(self):  # 定义未知词元索引属性
        return self._unk  # 返回未知词元索引

    @property  # 定义属性装饰器
    def token_freqs(self):  # 定义词元频率属性
        return self.token_freqs  # 返回词元频率列表

# 定义统计语料库的函数
def count_corpus(tokens):  # 定义统计语料库的函数
    if len(tokens) == 0 or isinstance(tokens[0], list):  # 如果tokens为空或tokens的元素是列表
        tokens = [token for line in tokens for token in line]  # 展平tokens列表
    return collections.Counter(tokens)  # 返回词元计数器

# 定义截断或填充文本序列的函数
def truncate_pad(line, num_steps, padding_token):  # 定义截断或填充文本序列的函数
    """截断或填充文本序列"""
    if len(line) > num_steps:  # 如果序列长度超过num_steps
        return line[:num_steps]  # 截断到num_steps长度
    return line + [padding_token] * (num_steps - len(line))  # 填充到num_steps长度

# 定义将机器翻译的文本序列转换成小批量的函数
def build_array_nmt(lines, vocab, num_steps):  # 定义将机器翻译的文本序列转换成小批量的函数
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]  # 将词元列表转换为索引列表
    lines = [l + [vocab['<eos>']] for l in lines]  # 在每个序列末尾添加<eos>标记
    array = torch.tensor([truncate_pad(  # 将序列转换为张量
        l, num_steps, vocab['<pad>']) for l in lines])  # 对每个序列进行截断或填充
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)  # 计算每个序列的有效长度（不包括填充）
    return array, valid_len  # 返回张量和有效长度

# 定义加载数据迭代器的函数
def load_array(data_arrays, batch_size, is_train=True):  # 定义加载数据迭代器的函数
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)  # 创建数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 创建数据迭代器

# 定义返回翻译数据集的迭代器和词表的函数
def load_data_nmt(batch_size, num_steps, num_examples=600):  # 定义返回翻译数据集的迭代器和词表的函数
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())  # 读取并预处理数据
    source, target = tokenize_nmt(text, num_examples)  # 词元化数据
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])  # 创建源语言词表
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])  # 创建目标语言词表
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)  # 将源语言序列转换为张量
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)  # 将目标语言序列转换为张量
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)  # 组合所有数据数组
    data_iter = load_array(data_arrays, batch_size)  # 创建数据迭代器
    return data_iter, src_vocab, tgt_vocab  # 返回数据迭代器和词表

# ========== 从9_6.py复制的编码器-解码器基类 ==========

# 定义编码器类
class Encoder(nn.Module):  # 定义编码器类，继承自nn.Module
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):  # 初始化函数
        super(Encoder, self).__init__(**kwargs)  # 调用父类的初始化函数

    def forward(self, X, *args):  # 定义前向传播函数
        """前向传播函数"""
        raise NotImplementedError  # 抛出未实现错误，要求子类必须实现此方法

# 定义解码器类
class Decoder(nn.Module):  # 定义解码器类，继承自nn.Module
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):  # 初始化函数
        super(Decoder, self).__init__(**kwargs)  # 调用父类的初始化函数

    def init_state(self, enc_outputs, *args):  # 定义初始化状态的函数
        """初始化解码器的隐藏状态"""
        raise NotImplementedError  # 抛出未实现错误，要求子类必须实现此方法

    def forward(self, X, state):  # 定义前向传播函数
        """前向传播函数"""
        raise NotImplementedError  # 抛出未实现错误，要求子类必须实现此方法

# 定义编码器-解码器类
class EncoderDecoder(nn.Module):  # 定义编码器-解码器类，继承自nn.Module
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):  # 初始化函数
        super(EncoderDecoder, self).__init__(**kwargs)  # 调用父类的初始化函数
        self.encoder = encoder  # 保存编码器
        self.decoder = decoder  # 保存解码器

    def forward(self, enc_X, dec_X, *args):  # 定义前向传播函数
        """前向传播函数"""
        # 编码器处理输入序列
        enc_outputs = self.encoder(enc_X, *args)  # 将编码器输入传递给编码器，得到编码器输出
        # 初始化解码器的状态
        dec_state = self.decoder.init_state(enc_outputs, *args)  # 使用编码器输出初始化解码器状态
        # 解码器处理输出序列
        return self.decoder(dec_X, dec_state)  # 将解码器输入和状态传递给解码器，得到最终输出

# ========== 工具类 ==========

# 定义计时器类
class Timer:  # 定义计时器类
    def __init__(self):  # 初始化函数
        self.times = []  # 存储每次运行的时间
        self.start()  # 启动计时器
    
    def start(self):  # 启动计时器函数
        self.tik = time.time()  # 记录当前时间
    
    def stop(self):  # 停止计时器函数
        self.times.append(time.time() - self.tik)  # 记录运行时间
        return self.times[-1]  # 返回最后一次运行时间
    
    def avg(self):  # 计算平均时间函数
        return sum(self.times) / len(self.times)  # 返回平均时间
    
    def sum(self):  # 计算总时间函数
        return sum(self.times)  # 返回总时间
    
    def cumsum(self):  # 计算累积时间函数
        return [sum(self.times[:i+1]) for i in range(len(self.times))]  # 返回累积时间列表

# 定义累加器类
class Accumulator:  # 定义累加器类
    def __init__(self, n):  # 初始化函数，n为需要累加的变量数量
        self.data = [0.0] * n  # 初始化n个累加变量为0.0
    
    def add(self, *args):  # 添加函数，用于累加多个值
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # 将每个累加变量加上对应的值
    
    def reset(self):  # 重置函数
        self.data = [0.0] * len(self.data)  # 将所有累加变量重置为0.0
    
    def __getitem__(self, idx):  # 索引函数，用于获取指定索引的累加变量
        return self.data[idx]  # 返回指定索引的累加变量

# 定义动画绘制类
class Animator:  # 定义动画绘制类
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):  # 初始化函数
        if legend is None:  # 如果没有指定图例
            legend = []  # 初始化图例为空列表
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)  # 创建子图
        if nrows * ncols == 1:  # 如果只有一个子图
            self.axes = [self.axes]  # 将axes转换为列表
        self.config_axes = lambda: self.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)  # 配置坐标轴
        self.X, self.Y, self.fmts = None, None, fmts  # 初始化X、Y和格式
    
    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):  # 设置坐标轴函数
        ax.set_xlabel(xlabel)  # 设置x轴标签
        ax.set_ylabel(ylabel)  # 设置y轴标签
        ax.set_xscale(xscale)  # 设置x轴刻度
        ax.set_yscale(yscale)  # 设置y轴刻度
        ax.set_xlim(xlim)  # 设置x轴范围
        ax.set_ylim(ylim)  # 设置y轴范围
        if legend:  # 如果有图例
            ax.legend(legend)  # 添加图例
        ax.grid()  # 添加网格
    
    def add(self, x, y):  # 添加数据点函数
        if not hasattr(y, "__len__"):  # 如果y不是列表或数组
            y = [y]  # 将y转换为列表
        n = len(y)  # 获取y的长度
        if not hasattr(x, "__len__"):  # 如果x不是列表或数组
            x = [x] * n  # 将x重复n次
        if not self.X:  # 如果X为空
            self.X = [[] for _ in range(n)]  # 初始化X为n个空列表
        if not self.Y:  # 如果Y为空
            self.Y = [[] for _ in range(n)]  # 初始化Y为n个空列表
        for i, (a, b) in enumerate(zip(x, y)):  # 遍历x和y的每个元素
            if a is not None and b is not None:  # 如果a和b都不为None
                self.X[i].append(a)  # 将a添加到对应的X列表
                self.Y[i].append(b)  # 将b添加到对应的Y列表
        self.axes[0].cla()  # 清除当前子图
        for x, y, fmt in zip(self.X, self.Y, self.fmts):  # 遍历X、Y和格式
            self.axes[0].plot(x, y, fmt)  # 绘制曲线
        self.config_axes()  # 配置坐标轴
        plt.draw()  # 绘制图形
        plt.pause(0.001)  # 暂停一小段时间

# 定义尝试使用GPU的函数
def try_gpu(i=0):  # 定义尝试使用GPU的函数
    if torch.cuda.device_count() >= i + 1:  # 如果GPU数量大于等于i+1
        return torch.device(f'cuda:{i}')  # 返回第i个GPU设备
    else:  # 如果GPU数量不足
        return torch.device('cpu')  # 返回CPU设备

# 定义梯度截断函数
def clip_grad(net, theta):  # 定义梯度截断函数
    if isinstance(net, nn.Module):  # 如果是PyTorch模型
        params = [p for p in net.parameters() if p.requires_grad]  # 获取所有需要梯度的参数
    else:  # 如果是自定义模型
        params = net.params  # 获取模型参数
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))  # 计算梯度的L2范数
    if norm > theta:  # 如果梯度范数大于阈值theta
        for param in params:  # 遍历所有参数
            param.grad[:] *= theta / norm  # 将梯度缩放到阈值theta

# ========== 9.7.1 编码器 ==========

# 定义序列到序列学习的循环神经网络编码器类
class Seq2SeqEncoder(Encoder):  # 定义Seq2SeqEncoder类，继承自Encoder
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):  # 初始化函数
        super(Seq2SeqEncoder, self).__init__(**kwargs)  # 调用父类的初始化函数
        # 嵌入层：将词元索引转换为嵌入向量
        self.embedding = nn.Embedding(vocab_size, embed_size)  # 创建嵌入层，输入大小为词表大小，输出大小为嵌入维度
        # GRU层：处理序列信息
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)  # 创建GRU层，输入大小为嵌入维度，隐藏层大小为num_hiddens，层数为num_layers

    def forward(self, X, *args):  # 定义前向传播函数
        # 输出'X'的形状：(batch_size, num_steps, embed_size)
        X = self.embedding(X)  # 将输入词元索引转换为嵌入向量
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)  # 交换维度：(num_steps, batch_size, embed_size)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)  # 通过GRU层，得到输出和最终隐藏状态
        # output的形状:(num_steps, batch_size, num_hiddens)
        # state的形状:(num_layers, batch_size, num_hiddens)
        return output, state  # 返回输出和状态

# 测试编码器
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)  # 创建编码器实例
encoder.eval()  # 设置为评估模式
X = torch.zeros((4, 7), dtype=torch.long)  # 创建输入张量，形状为(4, 7)
output, state = encoder(X)  # 前向传播
print("output.shape:", output.shape)  # 打印输出形状
print("state.shape:", state.shape)  # 打印状态形状

# ========== 9.7.2 解码器 ==========

# 定义序列到序列学习的循环神经网络解码器类
class Seq2SeqDecoder(Decoder):  # 定义Seq2SeqDecoder类，继承自Decoder
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):  # 初始化函数
        super(Seq2SeqDecoder, self).__init__(**kwargs)  # 调用父类的初始化函数
        # 嵌入层：将词元索引转换为嵌入向量
        self.embedding = nn.Embedding(vocab_size, embed_size)  # 创建嵌入层
        # GRU层：处理序列信息，输入是嵌入向量和上下文向量的拼接
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)  # 创建GRU层，输入大小为嵌入维度+隐藏层维度
        # 全连接层：将隐藏状态映射到词表
        self.dense = nn.Linear(num_hiddens, vocab_size)  # 创建线性层，将隐藏状态映射到词表大小

    def init_state(self, enc_outputs, *args):  # 定义初始化状态的函数
        return enc_outputs[1]  # 返回编码器的最终隐藏状态作为解码器的初始状态

    def forward(self, X, state):  # 定义前向传播函数
        # 输出'X'的形状：(batch_size, num_steps, embed_size)
        X = self.embedding(X).permute(1, 0, 2)  # 将输入词元索引转换为嵌入向量，并交换维度
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)  # 将最后一层的隐藏状态重复num_steps次，作为上下文
        X_and_context = torch.cat((X, context), 2)  # 将嵌入向量和上下文向量拼接
        output, state = self.rnn(X_and_context, state)  # 通过GRU层，得到输出和新状态
        output = self.dense(output).permute(1, 0, 2)  # 通过全连接层，并交换维度
        # output的形状:(batch_size, num_steps, vocab_size)
        return output, state  # 返回输出和新状态

# 测试解码器
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)  # 创建解码器实例
decoder.eval()  # 设置为评估模式
state = decoder.init_state(encoder(X))  # 使用编码器的输出初始化解码器状态
output, state = decoder(X, state)  # 前向传播
print("output.shape:", output.shape)  # 打印输出形状
print("state.shape:", state.shape)  # 打印状态形状

# ========== 9.7.3 损失函数 ==========

# 定义在序列中屏蔽不相关项的函数
def sequence_mask(X, valid_len, value=0):  # 定义序列掩码函数
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)  # 获取序列的最大长度
    mask = torch.arange((maxlen), dtype=torch.float32,
                         device=X.device)[None, :] < valid_len[:, None]  # 创建掩码：比较每个位置是否小于有效长度
    X[~mask] = value  # 将掩码为False的位置设置为指定值
    return X  # 返回掩码后的张量

# 测试序列掩码
X = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 创建测试张量
print("sequence_mask(X, torch.tensor([1, 2]))", sequence_mask(X, torch.tensor([1, 2])))  # 测试并打印结果

X = torch.ones((2, 3, 4))  # 创建测试张量
print("sequence_mask(X, torch.tensor([1, 2]))", sequence_mask(X, torch.tensor([1, 2])))  # 测试并打印结果

# 定义带遮蔽的softmax交叉熵损失函数类
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):  # 定义MaskedSoftmaxCELoss类，继承自nn.CrossEntropyLoss
    """带遮蔽的softmax交叉熵损失函数"""
    def forward(self, pred, label, valid_len):  # 定义前向传播函数
        weights = torch.ones_like(label)  # 创建权重张量，初始全为1
        weights = sequence_mask(weights, valid_len)  # 应用序列掩码，填充位置权重为0
        self.reduction = 'none'  # 设置不进行归约，返回每个位置的损失
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)  # 计算交叉熵损失
        weighted_loss = (unweighted_loss * weights).mean(dim=1)  # 应用权重并计算平均
        return weighted_loss  # 返回加权损失

# 测试损失函数
loss = MaskedSoftmaxCELoss()  # 创建损失函数实例
loss(torch.ones((3, 4, 10)), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0]))  # 测试损失函数

# ========== 9.7.4 训练 ==========

# 定义训练序列到序列模型的函数
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):  # 定义训练函数
    """训练序列到序列模型"""
    def xavier_init_weights(m):  # 定义Xavier初始化的内部函数
        if type(m) == nn.Linear:  # 如果是线性层
            nn.init.xavier_uniform_(m.weight)  # 使用Xavier均匀分布初始化权重
        if type(m) == nn.GRU:  # 如果是GRU层
            for param in m._flat_weights_names:  # 遍历所有参数名称
                if "weight" in param:  # 如果是权重参数
                    nn.init.xavier_uniform_(m._parameters[param])  # 使用Xavier均匀分布初始化权重
    net.apply(xavier_init_weights)  # 应用Xavier初始化到所有层
    net.to(device)  # 将模型移动到指定设备
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 创建Adam优化器
    loss = MaskedSoftmaxCELoss()  # 创建带掩码的交叉熵损失函数
    net.train()  # 设置为训练模式
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])  # 创建动画绘制对象
    for epoch in range(num_epochs):  # 遍历每个epoch
        timer = Timer()  # 创建计时器
        metric = Accumulator(2)  # 创建累加器，用于累加训练损失和词元数量
        for batch in data_iter:  # 遍历数据迭代器
            optimizer.zero_grad()  # 清零梯度
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]  # 将批次数据移动到指定设备
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)  # 创建开始标记张量
            dec_X = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学：使用真实标签作为输入（除了最后一个）
            Y_hat, _ = net(X, dec_X, X_valid_len)  # 前向传播
            l = loss(Y_hat, Y, Y_valid_len)  # 计算损失
            l.sum().backward()  # 反向传播，计算梯度
            clip_grad(net, 1)  # 梯度截断
            num_tokens = Y_valid_len.sum()  # 计算有效词元数量
            optimizer.step()  # 更新参数
            with torch.no_grad():  # 在不计算梯度的上下文中执行
                metric.add(l.sum(), num_tokens)  # 累加损失和词元数量
        if (epoch + 1) % 10 == 0:  # 每训练10个epoch
            animator.add(epoch + 1, (metric[0] / metric[1],))  # 更新动画

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')  # 打印最终损失和速度

# 设置模型超参数
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1  # 设置嵌入维度、隐藏层大小、层数和dropout率
batch_size, num_steps = 64, 10  # 设置批量大小和时间步数
lr, num_epochs, device = 0.005, 300, try_gpu()  # 设置学习率、训练轮数和设备

# 加载数据
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)  # 加载训练数据迭代器和词表

# 创建编码器和解码器
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)  # 创建编码器
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)  # 创建解码器
net = EncoderDecoder(encoder, decoder)  # 创建编码器-解码器模型

# 训练模型
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)  # 训练序列到序列模型

# ========== 9.7.5 预测 ==========

# 定义序列到序列模型的预测函数
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):  # 定义预测函数
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()  # 设置为评估模式
    src_tokens = src_vocab[src_sentence.lower().split()] + [src_vocab['<eos>']]  # 将源句子转换为词元索引，并添加结束标记
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)  # 计算源序列的有效长度
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])  # 截断或填充源序列
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)  # 添加批量维度
    enc_outputs = net.encoder(enc_X, enc_valid_len)  # 编码器前向传播
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)  # 初始化解码器状态
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)  # 创建解码器输入（开始标记）
    output_seq, attention_seq = [], []  # 初始化输出序列和注意力序列
    for _ in range(num_steps):  # 遍历每个时间步
        Y, dec_state = net.decoder(dec_X, dec_state)  # 解码器前向传播
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)  # 获取预测概率最大的词元索引
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()  # 去除批量维度并转换为整数
        # 保存注意力权重（稍后讨论）
        if 'attention_weights' in dir(net.decoder):  # 如果解码器有注意力权重
            attention_seq.append(net.decoder.attention_weights)  # 保存注意力权重
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:  # 如果预测到结束标记
            break  # 跳出循环
        # 保存解码输出
        output_seq.append(pred)  # 将预测词元添加到输出序列
        dec_X = torch.unsqueeze(dec_X, dim=0)  # 添加批量维度，作为下一时间步的输入
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_seq  # 返回预测序列和注意力序列

# ========== 9.7.6 预测序列的评估 ==========

# 定义计算BLEU分数的函数
def bleu(pred_seq, label_seq, k):  # 定义BLEU函数
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(), label_seq.split()  # 将预测序列和标签序列分割为词元
    len_pred, len_label = len(pred_tokens), len(label_tokens)  # 获取预测序列和标签序列的长度
    score = math.exp(min(0, 1 - len_label / len_pred))  # 计算长度惩罚因子
    for n in range(1, k + 1):  # 遍历n-gram，n从1到k
        num_matches, label_subs = 0, collections.defaultdict(int)  # 初始化匹配数和标签子串计数器
        for i in range(len_label - n + 1):  # 遍历标签序列的所有n-gram
            label_subs[' '.join(label_tokens[i: i + n])] += 1  # 统计每个n-gram的出现次数
        for i in range(len_pred - n + 1):  # 遍历预测序列的所有n-gram
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:  # 如果预测的n-gram在标签中存在
                num_matches += 1  # 增加匹配数
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1  # 减少计数，避免重复匹配
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))  # 计算n-gram精度并累乘
    return score  # 返回BLEU分数

# 测试翻译
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']  # 英语测试句子
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']  # 法语真实翻译

for eng, fra in zip(engs, fras):  # 遍历测试句子
    translation, attention_seq = predict_seq2seq(  # 预测翻译
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')  # 打印翻译结果和BLEU分数
