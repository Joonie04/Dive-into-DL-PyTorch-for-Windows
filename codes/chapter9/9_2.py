# 9.2 长短期记忆网络（LSTM）

import math  # 导入math模块，用于数学计算
import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
from torch.nn import functional as F  # 导入PyTorch函数式接口
import collections  # 导入collections模块，用于计数
import re  # 导入re模块，用于正则表达式
import os  # 导入os模块，用于文件路径操作
import sys  # 导入sys模块，用于系统路径操作
import time  # 导入time模块，用于计时
import random  # 导入random模块，用于随机操作
from matplotlib import pyplot as plt  # 导入matplotlib.pyplot绘图模块

# 添加downloader目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'downloader'))  # 将downloader目录添加到Python路径

from time_machine import download  # 导入下载函数

# 8.2.1 读取数据集
# 定义读取时间机器数据集的函数
def read_time_machine():  # 定义读取时间机器数据集的函数
    file_path = download('time_machine')  # 下载时间机器数据集
    with open(file_path, 'r', encoding='utf-8') as f:  # 以只读模式打开文件，使用UTF-8编码
        lines = f.readlines()  # 读取所有行
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]  # 对每行进行处理：移除非字母字符，转换为小写

# 8.2.2 词元化
# 定义词元化函数
def tokenize(lines, token='word'):  # 定义词元化函数
    if token == 'word':  # 如果词元类型是单词
        return [line.split() for line in lines]  # 按空格分割每行，返回单词列表
    elif token == 'char':  # 如果词元类型是字符
        return [list(line) for line in lines]  # 将每行转换为字符列表
    else:  # 如果词元类型未知
        print('错误：未知词元类型：' + token)  # 打印错误信息

# 8.2.3 词表
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

# 8.2.4 整合所有函数
# 定义加载时间机器语料库的函数
def load_corpus_time_machine(max_tokens=-1):  # 定义加载时间机器语料库的函数
    lines = read_time_machine()  # 读取时间机器数据集
    tokens = tokenize(lines, 'char')  # 按字符进行词元化
    vocab = Vocab(tokens)  # 创建词表
    corpus = [vocab[token] for line in tokens for token in line]  # 将词元转换为索引，展平为语料库
    if max_tokens > 0:  # 如果指定了最大词元数
        corpus = corpus[:max_tokens]  # 截取前max_tokens个词元
    return corpus, vocab  # 返回语料库和词表

# 定义尝试使用GPU的函数
def try_gpu(i=0):  # 定义尝试使用GPU的函数
    if torch.cuda.device_count() >= i + 1:  # 如果GPU数量大于等于i+1
        return torch.device(f'cuda:{i}')  # 返回第i个GPU设备
    else:  # 如果GPU数量不足
        return torch.device('cpu')  # 返回CPU设备

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

# 定义SGD优化器函数
def sgd(params, lr, batch_size):  # 定义随机梯度下降优化器函数
    with torch.no_grad():  # 在不计算梯度的上下文中执行
        for param in params:  # 遍历所有参数
            param -= lr * param.grad / batch_size  # 使用梯度更新参数：param = param - lr * grad / batch_size

# 定义随机抽样序列数据迭代器函数
def seq_data_iter_random(corpus, batch_size, num_steps):  # 定义随机抽样序列数据迭代器函数
    corpus = corpus[random.randint(0, num_steps - 1):]  # 随机截取语料库，从0到num_steps-1之间的随机位置开始
    num_subseqs = (len(corpus) - 1) // num_steps  # 计算子序列的数量
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))  # 生成子序列的起始索引列表
    random.shuffle(initial_indices)  # 随机打乱起始索引

    def data(pos):  # 定义数据提取函数
        return corpus[pos: pos + num_steps]  # 返回从pos开始的num_steps个词元

    num_batches = num_subseqs // batch_size  # 计算批次数
    for i in range(0, batch_size * num_batches, batch_size):  # 遍历每个批次
        initial_indices_per_batch = initial_indices[i: i + batch_size]  # 获取当前批次的起始索引
        X = [data(j) for j in initial_indices_per_batch]  # 提取输入序列
        Y = [data(j + 1) for j in initial_indices_per_batch]  # 提取目标序列（输入序列的下一个词元）
        yield torch.tensor(X), torch.tensor(Y)  # 生成批次数据

# 定义顺序分区序列数据迭代器函数
def seq_data_iter_sequential(corpus, batch_size, num_steps):  # 定义顺序分区序列数据迭代器函数
    offset = random.randint(0, num_steps)  # 随机选择偏移量，用于打乱批次之间的顺序
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size  # 计算可以使用的词元数量
    Xs = torch.tensor(corpus[offset: offset + num_tokens])  # 提取输入序列
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])  # 提取目标序列（输入序列的下一个词元）
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)  # 重塑为(batch_size, -1)形状
    num_batches = Xs.shape[1] // num_steps  # 计算批次数
    for i in range(0, num_steps * num_batches, num_steps):  # 遍历每个批次
        X = Xs[:, i: i + num_steps]  # 提取当前批次的输入序列
        Y = Ys[:, i: i + num_steps]  # 提取当前批次的目标序列
        yield X, Y  # 生成批次数据

# 定义加载数据的函数
def load_data_time_machine(batch_size, num_steps, use_random_iter=True, max_tokens=10000):  # 定义加载时间机器数据的函数
    lines = read_time_machine()  # 读取时间机器数据集
    tokens = tokenize(lines, 'char')  # 按字符进行词元化
    vocab = Vocab(tokens)  # 创建词表
    corpus = [vocab[token] for line in tokens for token in line]  # 将词元转换为索引，展平为语料库
    if max_tokens > 0:  # 如果指定了最大词元数
        corpus = corpus[:max_tokens]  # 截取前max_tokens个词元
    
    if use_random_iter:  # 如果使用随机迭代器
        data_iter_fn = seq_data_iter_random  # 设置为随机抽样迭代器
    else:  # 如果使用顺序迭代器
        data_iter_fn = seq_data_iter_sequential  # 设置为顺序分区迭代器
    
    data_iter = data_iter_fn(corpus, batch_size, num_steps)  # 创建数据迭代器
    return data_iter, vocab  # 返回数据迭代器和词表

# 定义预测函数
def predict_ch8(prefix, num_preds, net, vocab, device):  # 定义预测函数
    state = net.begin_state(batch_size=1, device=device)  # 初始化状态
    outputs = [vocab[prefix[0]]]  # 初始化输出列表，包含前缀的第一个字符
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))  # 定义获取输入的函数
    for y in prefix[1:]:  # 遍历前缀的剩余字符
        _, state = net(get_input(), state)  # 前向传播
        outputs.append(vocab[y])  # 将字符索引添加到输出列表
    for _ in range(num_preds):  # 预测num_preds个字符
        y, state = net(get_input(), state)  # 前向传播
        outputs.append(int(y.argmax(dim=1).reshape(1)))  # 将预测结果（最大概率的索引）添加到输出列表
    return ''.join([vocab.idx_to_token[i] for i in outputs])  # 将索引转换为字符并拼接成字符串

# 定义梯度截断函数
def grad_clipping(net, theta):  # 定义梯度截断函数
    if isinstance(net, nn.Module):  # 如果是PyTorch模型
        params = [p for p in net.parameters() if p.requires_grad]  # 获取所有需要梯度的参数
    else:  # 如果是自定义模型
        params = net.params  # 获取模型参数
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))  # 计算梯度的L2范数
    if norm > theta:  # 如果梯度范数大于阈值theta
        for param in params:  # 遍历所有参数
            param.grad[:] *= theta / norm  # 将梯度缩放到阈值theta

# 定义训练一个epoch的函数
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter=False):  # 定义训练一个epoch的函数
    state, timer = None, Timer()  # 初始化状态和计时器
    metric = Accumulator(2)  # 创建累加器，用于累加训练损失和词元数量
    for X, Y in train_iter:  # 遍历训练数据迭代器
        if state is None or use_random_iter:  # 如果是第一次迭代或使用随机抽样
            state = net.begin_state(batch_size=X.shape[0], device=device)  # 初始化状态
        else:  # 如果不是第一次迭代且使用顺序分区
            for s in state:  # 遍历状态
                s.detach_()  # 分离状态，不计算梯度
        y = Y.T.reshape(-1)  # 重塑目标标签
        X, y = X.to(device), y.to(device)  # 将输入和标签移动到指定设备
        y_hat, state = net(X, state)  # 前向传播
        l = loss(y_hat, y.long()).mean()  # 计算损失
        if isinstance(updater, torch.optim.Optimizer):  # 如果使用PyTorch优化器
            updater.zero_grad()  # 清零梯度
            l.backward()  # 反向传播
            grad_clipping(net, 1)  # 梯度截断
            updater.step()  # 更新参数
        else:  # 如果使用自定义SGD优化器
            l.backward()  # 反向传播
            grad_clipping(net, 1)  # 梯度截断
            updater(batch_size=1)  # 更新参数
        metric.add(l * y.numel(), y.numel())  # 累加训练损失和词元数量
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()  # 返回困惑度和每秒处理的词元数

# 定义训练函数
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):  # 定义训练函数
    loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    animator = Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])  # 创建动画绘制对象
    # 初始化优化器
    if isinstance(net, nn.Module):  # 如果是PyTorch模型
        updater = torch.optim.Adam(net.parameters(), lr)  # 使用Adam优化器
    else:  # 如果是自定义模型
        updater = lambda batch_size: sgd(net.params, lr, batch_size)  # 使用SGD优化器
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)  # 定义预测函数
    # 训练和预测
    for epoch in range(num_epochs):  # 遍历每个epoch
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)  # 训练一个epoch
        if (epoch + 1) % 10 == 0:  # 每训练10个epoch
            print(predict('time traveller'))  # 打印预测结果
            animator.add(epoch + 1, [ppl])  # 更新动画
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')  # 打印最终困惑度和速度
    print(predict('time traveller'))  # 打印预测结果
    print(predict('traveller'))  # 打印预测结果

# 定义RNN模型类（从零实现）
class RNNModelScratch:  # 定义RNN模型类
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):  # 初始化函数
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens  # 保存词表大小和隐藏层大小
        self.params = get_params(vocab_size, num_hiddens, device)  # 初始化模型参数
        self.init_state, self.forward_fn = init_state, forward_fn  # 保存状态初始化函数和前向传播函数

    def __call__(self, X, state):  # 定义调用函数
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)  # 对输入进行独热编码，并转换为float32类型
        return self.forward_fn(X, state, self.params)  # 调用前向传播函数

    def begin_state(self, batch_size, device):  # 定义开始状态的函数
        return self.init_state(batch_size, self.num_hiddens, device)  # 返回初始状态

# 定义RNN模型类（简洁实现）
class RNNModel(nn.Module):  # 定义RNN模型类，继承自nn.Module
    def __init__(self, rnn_layer, vocab_size, **kwargs):  # 初始化函数
        super(RNNModel, self).__init__(**kwargs)  # 调用父类的初始化函数
        self.rnn = rnn_layer  # 保存RNN层
        self.vocab_size = vocab_size  # 保存词表大小
        self.num_hiddens = self.rnn.hidden_size  # 获取隐藏层大小
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:  # 如果RNN不是双向的
            self.num_directions = 1  # 方向数为1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)  # 创建线性层，将隐藏状态映射到词表大小
        else:  # 如果RNN是双向的
            self.num_directions = 2  # 方向数为2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)  # 创建线性层，将双向隐藏状态映射到词表大小

    def forward(self, inputs, state):  # 定义前向传播函数
        # inputs的形状：(时间步数量, 批量大小, 词表大小)
        X = F.one_hot(inputs.T.long(), self.vocab_size)  # 对输入进行独热编码
        X = X.to(torch.float32)  # 转换为float32类型
        Y, state = self.rnn(X, state)  # 通过RNN层
        # 全连接层首先将Y的形状改为(时间步数量*批量大小, 隐藏单元数量)
        # 它的输出形状是(时间步数量*批量大小, 词表大小)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))  # 重塑Y并通过线性层
        return output, state  # 返回输出和新状态

    def begin_state(self, device, batch_size=1):  # 定义初始化状态的函数
        if not isinstance(self.rnn, nn.LSTM):  # 如果不是LSTM
            # nn.GRU以张量作为隐藏状态
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)  # 返回全零张量
        else:  # 如果是LSTM
            # nn.LSTM以元组作为隐藏状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),  # 返回隐藏状态元组
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))

# 9.2.2 从零开始实现
# 加载数据
batch_size, num_steps = 32, 35  # 设置批量大小为32，时间步数为35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)  # 加载时间机器数据

# 定义初始化LSTM参数的函数
def get_lstm_params(vocab_size, num_hiddens, device):  # 定义初始化LSTM参数的函数
    num_inputs = num_outputs = vocab_size  # 输入维度和输出维度都等于词表大小

    def normal(shape):  # 定义生成正态分布随机数的函数
        return torch.randn(size=shape, device=device) * 0.01  # 返回标准正态分布随机数，乘以0.01

    def three():  # 定义生成三个参数的函数（权重、权重、偏置）
        return (normal((num_inputs, num_hiddens)),  # 输入到隐藏状态的权重矩阵
                normal((num_hiddens, num_hiddens)),  # 隐藏状态到隐藏状态的权重矩阵
                torch.zeros(num_hiddens, device=device))  # 偏置向量

    # 输入门参数：控制信息流入记忆细胞的程度
    W_xi, W_hi, b_i = three()  # 输入门：W_xi, W_hi, b_i
    # 遗忘门参数：控制遗忘前一时刻记忆的程度
    W_xf, W_hf, b_f = three()  # 遗忘门：W_xf, W_hf, b_f
    # 输出门参数：控制记忆细胞输出到隐藏状态的程度
    W_xo, W_ho, b_o = three()  # 输出门：W_xo, W_ho, b_o
    # 候选记忆细胞参数：用于计算候选记忆细胞
    W_xc, W_hc, b_c = three()  # 候选记忆细胞：W_xc, W_hc, b_c

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))  # 隐藏状态到输出的权重矩阵，形状为(num_hiddens, num_outputs)
    b_q = torch.zeros(num_outputs, device=device)  # 输出层的偏置，形状为(num_outputs)

    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]  # 将所有参数组合成列表
    for param in params:  # 遍历所有参数
        param.requires_grad_(True)  # 启用梯度计算
    return params  # 返回参数列表

# 定义初始化LSTM状态的函数
def init_lstm_state(batch_size, num_hiddens, device):  # 定义初始化LSTM状态的函数
    return (torch.zeros((batch_size, num_hiddens), device=device),  # 返回初始隐藏状态，形状为(batch_size, num_hiddens)
            torch.zeros((batch_size, num_hiddens), device=device))  # 返回初始记忆细胞，形状为(batch_size, num_hiddens)

# 定义LSTM前向传播函数
def lstm(inputs, state, params):  # 定义LSTM前向传播函数
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params  # 解包参数
    (H, C) = state  # 解包状态，H为隐藏状态，C为记忆细胞
    outputs = []  # 初始化输出列表
    for X in inputs:  # 遍历每个时间步的输入
        # 计算输入门：控制信息流入记忆细胞的程度
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)  # 输入门：σ(X*W_xi + H*W_hi + b_i)
        # 计算遗忘门：控制遗忘前一时刻记忆的程度
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)  # 遗忘门：σ(X*W_xf + H*W_hf + b_f)
        # 计算输出门：控制记忆细胞输出到隐藏状态的程度
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)  # 输出门：σ(X*W_xo + H*W_ho + b_o)
        # 计算候选记忆细胞：用于更新记忆细胞
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)  # 候选记忆细胞：tanh(X*W_xc + H*W_hc + b_c)
        # 更新记忆细胞：使用遗忘门和输入门
        C = F * C + I * C_tilda  # 新记忆细胞：F*C + I*C_tilda
        # 计算新的隐藏状态：使用输出门控制记忆细胞的输出
        H = O * torch.tanh(C)  # 新隐藏状态：O*tanh(C)
        # 计算输出
        Y = (H @ W_hq) + b_q  # 输出：H*W_hq + b_q
        outputs.append(Y)  # 将输出添加到输出列表
    return torch.cat(outputs, dim=0), (H, C)  # 返回所有输出和新的状态（隐藏状态和记忆细胞）

# 训练LSTM模型（从零实现）
vocab_size, num_hiddens, device = len(vocab), 256, try_gpu()  # 设置词表大小、隐藏层大小和设备
model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)  # 创建LSTM模型
num_epochs, lr = 500, 1  # 设置训练轮数为500，学习率为1
train_ch8(model, train_iter, vocab, lr, num_epochs, device)  # 训练LSTM模型

# 9.2.3 简洁实现
# 使用PyTorch内置的LSTM层
num_inputs = vocab_size  # 输入维度等于词表大小
lstm_layer = nn.LSTM(num_inputs, num_hiddens)  # 创建LSTM层，输入维度为词表大小，隐藏层大小为256
model = RNNModel(lstm_layer, len(vocab))  # 创建LSTM模型
model = model.to(device)  # 将模型移动到指定设备
train_ch8(model, train_iter, vocab, lr, num_epochs, device)  # 训练LSTM模型
