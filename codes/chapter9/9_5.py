# 9.5 机器翻译与数据集

import os  # 导入os模块，用于文件路径操作
import sys  # 导入sys模块，用于系统路径操作
import torch  # 导入PyTorch库
from torch.utils import data  # 导入PyTorch数据工具
import collections  # 导入collections模块，用于计数
from matplotlib import pyplot as plt  # 导入matplotlib.pyplot绘图模块

# 添加downloader目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'downloader'))  # 将downloader目录添加到Python路径

from fra_eng import download  # 导入下载函数

# 9.5.1 下载和预处理数据集
# 定义读取"英语-法语"数据集的函数
def read_data_nmt():  # 定义读取"英语-法语"数据集的函数
    """载入"英语-法语"数据集"""
    data_dir = download('fra-eng')  # 下载并解压"fra-eng"数据集
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:  # 以只读模式打开fra.txt文件，使用UTF-8编码
        return f.read()  # 读取并返回文件内容

# 读取原始文本数据
raw_text = read_data_nmt()  # 调用read_data_nmt函数读取数据
print("raw_text[:75]:", raw_text[:75])  # 打印原始文本的前75个字符

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

# 预处理文本
text = preprocess_nmt(raw_text)  # 调用preprocess_nmt函数预处理文本
print("text[:80]:", text[:80])  # 打印预处理后文本的前80个字符

# 9.5.2 词元化
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

# 词元化文本
source, target = tokenize_nmt(text)  # 调用tokenize_nmt函数进行词元化
print("source[:6]:", source[:6])  # 打印前6个英语句子的词元
print("target[:6]:", target[:6])  # 打印前6个法语句子的词元

# 定义绘制列表长度对直方图的函数
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):  # 定义绘制列表长度对直方图的函数
    """绘制列表长度对的直方图"""
    plt.figure(figsize=(3.5, 2.5))  # 设置图形大小为3.5x2.5
    _, _, patches = plt.hist(  # 绘制直方图
        [[len(l) for l in xlist], [len(l) for l in ylist]])  # 绘制两个列表的长度分布
    plt.xlabel(xlabel)  # 设置x轴标签
    plt.ylabel(ylabel)  # 设置y轴标签
    for patch in patches[1].patches:  # 遍历第二个直方图的补丁
        patch.set_hatch('/')  # 设置斜线填充样式
    plt.legend(legend)  # 添加图例
    plt.show()  # 显示图形

# 绘制源语言和目标语言的序列长度分布
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence', 'count', source, target)  # 调用show_list_len_pair_hist函数绘制直方图

# 9.5.3 词表
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

# 创建源语言（英语）词表
src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])  # 创建词表，最小频率为2，保留特殊词元：<pad>（填充）、<bos>（开始）、<eos>（结束）
print("len(src_vocab):", len(src_vocab))  # 打印源语言词表的大小

# 9.5.4 加载数据集
# 定义截断或填充文本序列的函数
def truncate_pad(line, num_steps, padding_token):  # 定义截断或填充文本序列的函数
    """截断或填充文本序列"""
    if len(line) > num_steps:  # 如果序列长度超过num_steps
        return line[:num_steps]  # 截断到num_steps长度
    return line + [padding_token] * (num_steps - len(line))  # 填充到num_steps长度

# 测试截断或填充函数
print("truncate_pad(source[0], 10, src_vocab['<pad>']):", truncate_pad(source[0], 10, src_vocab['<pad>']))  # 打印截断或填充后的结果

# 定义将机器翻译的文本序列转换成小批量的函数
def build_array_nmt(lines, vocab, num_steps):  # 定义将机器翻译的文本序列转换成小批量的函数
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]  # 将词元列表转换为索引列表
    lines = [l + [vocab['<eos>']] for l in lines]  # 在每个序列末尾添加<eos>标记
    array = torch.tensor([truncate_pad(  # 将序列转换为张量
        l, num_steps, vocab['<pad>']) for l in lines])  # 对每个序列进行截断或填充
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)  # 计算每个序列的有效长度（不包括填充）
    return array, valid_len  # 返回张量和有效长度

# 9.5.5 训练模型
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

# 加载数据迭代器和词表
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)  # 加载数据，批量大小为2，时间步数为8
# 打印第一个批次的数据
for X, X_valid_len, Y, Y_valid_len in train_iter:  # 遍历数据迭代器
    print("X:", X)  # 打印源语言（英语）张量
    print("X_valid_len:", X_valid_len)  # 打印源语言序列的有效长度
    print("Y:", Y)  # 打印目标语言（法语）张量
    print("Y_valid_len:", Y_valid_len)  # 打印目标语言序列的有效长度
    break  # 只打印第一个批次
