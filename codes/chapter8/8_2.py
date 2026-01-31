# 8.2 文本预处理

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

# 读取数据集
lines = read_time_machine()  # 读取时间机器数据集
print(f'# 文本总行数: {len(lines)}')  # 打印文本总行数
print(lines[0])  # 打印第一行
print(lines[10])  # 打印第11行


# 8.2.2 词元化
# 定义词元化函数
def tokenize(lines, token='word'):  # 定义词元化函数
    if token == 'word':  # 如果词元类型是单词
        return [line.split() for line in lines]  # 按空格分割每行，返回单词列表
    elif token == 'char':  # 如果词元类型是字符
        return [list(line) for line in lines]  # 将每行转换为字符列表
    else:  # 如果词元类型未知
        print('错误：未知词元类型：' + token)  # 打印错误信息

# 进行词元化
tokens = tokenize(lines)  # 对文本进行词元化（默认按单词）
for i in range(11):  # 遍历前11行
    print(tokens[i])  # 打印每行的词元


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

# 创建词表
vocab = Vocab(tokens)  # 创建词表
print(list(vocab.token_to_idx.items())[:10])  # 打印前10个词元及其索引

# 测试词表
for i in [0, 10]:  # 遍历第1行和第11行
    print('文本:', tokens[i])  # 打印原始文本
    print('索引:', vocab[tokens[i]])  # 打印词元索引


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

# 加载语料库
corpus, vocab = load_corpus_time_machine()  # 加载时间机器语料库
print("len(corpus):", len(corpus))  # 打印语料库长度
print("len(vocab):", len(vocab))  # 打印词表大小
