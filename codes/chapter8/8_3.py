# 8.3.3 自然语言统计

import random  # 导入random模块，用于随机操作
import torch  # 导入PyTorch库
from matplotlib import pyplot as plt  # 导入matplotlib.pyplot绘图模块
import collections  # 导入collections模块，用于计数
import re  # 导入re模块，用于正则表达式
import os  # 导入os模块，用于文件路径操作
import sys  # 导入sys模块，用于系统路径操作

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

# 定义绘图函数
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='log', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):  # 定义绘图函数
    if legend is None:  # 如果没有指定图例
        legend = []  # 初始化图例为空列表
    plt.figure(figsize=figsize)  # 创建图形，设置大小
    if isinstance(X, list):  # 如果X是列表
        if Y is None:  # 如果没有指定Y
            for i, x in enumerate(X):  # 遍历X中的每个元素
                plt.plot(x, fmts[i % len(fmts)], label=legend[i] if i < len(legend) else None)  # 绘制曲线
        else:  # 如果指定了Y
            for i, (x, y) in enumerate(zip(X, Y)):  # 遍历X和Y中的每个元素
                plt.plot(x, y, fmts[i % len(fmts)], label=legend[i] if i < len(legend) else None)  # 绘制曲线
    else:  # 如果X不是列表
        if Y is None:  # 如果没有指定Y
            plt.plot(X, fmts[0], label=legend[0] if len(legend) > 0 else None)  # 绘制曲线
        else:  # 如果指定了Y
            plt.plot(X, Y, fmts[0], label=legend[0] if len(legend) > 0 else None)  # 绘制曲线
    plt.xlabel(xlabel)  # 设置x轴标签
    plt.ylabel(ylabel)  # 设置y轴标签
    plt.xscale(xscale)  # 设置x轴刻度
    plt.yscale(yscale)  # 设置y轴刻度
    plt.xlim(xlim)  # 设置x轴范围
    plt.ylim(ylim)  # 设置y轴范围
    if legend:  # 如果有图例
        plt.legend()  # 显示图例
    plt.grid()  # 添加网格
    plt.show()  # 显示图形

# 读取数据并进行词元化
tokens = tokenize(read_time_machine())  # 读取时间机器数据集并进行词元化（按单词）
corpus = [token for line in tokens for token in line]  # 展平词元列表，构建语料库
vocab = Vocab(corpus)  # 创建词表
print(vocab.token_freqs[:10])  # 打印前10个高频词元及其频率

# 绘制词元频率分布
freqs = [freq for token, freq in vocab.token_freqs]  # 提取词元频率列表
plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')  # 绘制词元频率分布（双对数坐标）

# 构建二元语法（bigram）词元
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]  # 构建二元语法：将相邻的两个词元组成对
bigram_vocab = Vocab(bigram_tokens)  # 创建二元语法词表
print(bigram_vocab.token_freqs[:10])  # 打印前10个高频二元语法及其频率

# 构建三元语法（trigram）词元
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]  # 构建三元语法：将相邻的三个词元组成三元组
trigram_vocab = Vocab(trigram_tokens)  # 创建三元语法词表
print(trigram_vocab.token_freqs[:10])  # 打印前10个高频三元语法及其频率

# 提取二元语法和三元语法的频率
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]  # 提取二元语法频率列表
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]  # 提取三元语法频率列表

# 绘制一元语法、二元语法和三元语法的频率分布对比
plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log', legend=['unigram', 'bigram', 'trigram'])  # 绘制三种语法的频率分布


# 8.3.4 读取长序列数据

## 1. 随机抽样
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

# 测试随机抽样序列数据迭代器
my_seq = list(range(35))  # 创建测试序列：0到34
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):  # 测试随机抽样迭代器
    print('X:', X, '\nY:', Y)  # 打印输入序列和目标序列

## 2. 顺序分区
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

# 测试顺序分区序列数据迭代器
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):  # 测试顺序分区迭代器
    print('X:', X, '\nY:', Y)  # 打印输入序列和目标序列

# 定义序列数据加载器类
class SeqDataLoader:  # 定义序列数据加载器类
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):  # 初始化函数
        if use_random_iter:  # 如果使用随机迭代器
            self.data_iter_fn = seq_data_iter_random  # 设置为随机抽样迭代器
        else:  # 如果使用顺序迭代器
            self.data_iter_fn = seq_data_iter_sequential  # 设置为顺序分区迭代器
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)  # 加载时间机器语料库
        self.batch_size, self.num_steps = batch_size, num_steps  # 设置批量大小和时间步数

    def __iter__(self):  # 定义迭代器函数
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)  # 返回数据迭代器

    def load_data_time_machine(batch_size, num_steps, use_random_iter=True, max_tokens=10000):  # 定义加载时间机器数据的函数
        data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)  # 创建序列数据加载器
        return data_iter, data_iter.vocab  # 返回数据迭代器和词表
