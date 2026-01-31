# 15.7 自然语言推断: 微调BERT
import json
import multiprocessing
import os
import re
import torch
from torch import nn
import math
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


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽无效位置
    
    参数:
        X: 输入张量
        valid_len: 有效长度
        value: 用于填充的值
    
    返回:
        屏蔽后的张量
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作
    
    参数:
        X: 输入张量
        valid_lens: 有效长度
    
    返回:
        softmax后的张量
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """缩放点积注意力
    
    使用缩放的点积来计算查询和键之间的注意力分数。
    
    参数:
        dropout: dropout率
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """前向传播
        
        参数:
            queries: 查询向量
            keys: 键向量
            values: 值向量
            valid_lens: 有效长度
        
        返回:
            注意力输出
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力
    
    将输入投影到多个子空间，并在每个子空间中独立计算注意力。
    
    参数:
        key_size: 键的维度
        query_size: 查询的维度
        value_size: 值的维度
        num_hiddens: 隐藏层维度
        num_heads: 注意力头的数量
        dropout: dropout率
        bias: 是否使用偏置
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """前向传播
        
        参数:
            queries: 查询向量
            keys: 键向量
            values: 值向量
            valid_lens: 有效长度
        
        返回:
            多头注意力输出
        """
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状
    
    参数:
        X: 输入张量
        num_heads: 注意力头的数量
    
    返回:
        变换后的张量
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作
    
    参数:
        X: 输入张量
        num_heads: 注意力头的数量
    
    返回:
        逆转后的张量
    """
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络
    
    对每个位置独立地应用相同的前馈网络。
    
    参数:
        num_hiddens: 隐藏层维度
        ffn_num_hiddens: 前馈网络隐藏层维度
        dropout: dropout率
    """
    def __init__(self, num_hiddens, ffn_num_hiddens, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(num_hiddens, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(ffn_num_hiddens, num_hiddens)
    
    def forward(self, X):
        """前向传播
        
        参数:
            X: 输入张量
        
        返回:
            前馈网络输出
        """
        return self.dense2(self.dropout(self.relu(self.dense1(X))))


class AddNorm(nn.Module):
    """残差连接和层规范化
    
    将残差连接和层规范化组合在一起。
    
    参数:
        normalized_shape: 规范化的形状
        dropout: dropout率
    """
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """前向传播
        
        参数:
            X: 输入张量
            Y: 要添加的张量
        
        返回:
            残差连接和层规范化后的输出
        """
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """Transformer编码器块
    
    包含多头注意力、残差连接、层规范化和前馈网络。
    
    参数:
        key_size: 键的维度
        query_size: 查询的维度
        value_size: 值的维度
        num_hiddens: 隐藏层维度
        norm_shape: 规范化的形状
        ffn_num_input: 前馈网络输入维度
        ffn_num_hiddens: 前馈网络隐藏层维度
        num_heads: 注意力头的数量
        dropout: dropout率
        bias: 是否使用偏置
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                            num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        """前向传播
        
        参数:
            X: 输入张量
            valid_lens: 有效长度
        
        返回:
            编码器块输出
        """
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class BERTEncoder(nn.Module):
    """BERT编码器
    
    BERT（Bidirectional Encoder Representations from Transformers）是一种预训练语言模型，
    使用Transformer编码器来学习双向的上下文表示。
    
    参数:
        vocab_size: 词表大小
        num_hiddens: 隐藏层维度
        norm_shape: 层规范化的形状
        ffn_num_input: 前馈网络输入维度
        ffn_num_hiddens: 前馈网络隐藏层维度
        num_heads: 注意力头的数量
        num_layers: 编码器层数
        dropout: dropout率
        max_len: 最大序列长度
        key_size: 键的维度
        query_size: 查询的维度
        value_size: 值的维度
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape, 
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens=None):
        """前向传播
        
        参数:
            tokens: 词元索引张量，形状为 (batch_size, num_steps)
            segments: 片段索引张量，形状为 (batch_size, num_steps)
            valid_lens: 有效长度（可选）
        
        返回:
            编码后的表示，形状为 (batch_size, num_steps, num_hiddens)
        """
        x = self.token_embedding(tokens) + self.segment_embedding(segments) + self.pos_embedding[:, :tokens.shape[1], :]
        for blk in self.blks:
            x = blk(x, valid_lens)
        return x


class MaskLM(nn.Module):
    """掩蔽语言模型任务
    
    在BERT预训练中，随机掩蔽输入序列中的一些词元，
    然后模型需要预测这些被掩蔽的词元。
    
    参数:
        vocab_size: 词表大小
        num_hiddens: 隐藏层维度
        num_inputs: 输入维度
    """
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Embedding(vocab_size, num_hiddens))
            

    def forward(self, X, pred_positions):
        """前向传播
        
        参数:
            X: 编码器输出，形状为 (batch_size, num_steps, num_hiddens)
            pred_positions: 预测位置，形状为 (batch_size, num_predictions)
        
        返回:
            预测结果，形状为 (batch_size, num_predictions, vocab_size)
        """
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """下一句预测任务
    
    在BERT预训练中，模型需要预测两个句子是否是连续的。
    
    参数:
        num_inputs: 输入维度
    """
    def __init__(self, num_inputs=768, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        """前向传播
        
        参数:
            X: 输入张量，形状为 (batch_size, num_hiddens)
        
        返回:
            预测结果，形状为 (batch_size, 2)
        """
        return self.output(X)


class BERTModel(nn.Module):
    """BERT模型
    
    整合了BERT编码器、掩蔽语言模型和下一句预测任务。
    
    参数:
        vocab_size: 词表大小
        num_hiddens: 隐藏层维度
        norm_shape: 层规范化的形状
        ffn_num_input: 前馈网络输入维度
        ffn_num_hiddens: 前馈网络隐藏层维度
        num_heads: 注意力头的数量
        num_layers: 编码器层数
        dropout: dropout率
        max_len: 最大序列长度
        key_size: 键的维度
        query_size: 查询的维度
        value_size: 值的维度
        hid_in_features: 隐藏层特征维度
        mlm_in_features: 掩蔽语言模型输入特征维度
        nsp_in_features: 下一句预测输入特征维度
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=max_len, key_size=key_size, query_size=query_size, value_size=value_size)
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)
        self.hidden = nn.Sequential(
            nn.Linear(hid_in_features, hid_in_features),
            nn.Tanh())

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        """前向传播
        
        参数:
            tokens: 词元索引张量
            segments: 片段索引张量
            valid_lens: 有效长度
            pred_positions: 预测位置（用于掩蔽语言模型）
        
        返回:
            encoded_X: 编码后的表示
            mlm_Y_hat: 掩蔽语言模型预测
            nsp_Y_hat: 下一句预测
        """
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU则返回CPU
    
    返回:
        设备列表，包含所有可用的GPU或CPU
    """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


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


def get_tokens_and_segments(tokens_a, tokens_b):
    """获取BERT输入的词元和片段索引
    
    参数:
        tokens_a: 第一个句子的词元列表
        tokens_b: 第二个句子的词元列表
    
    返回:
        (tokens, segments) 元组
        - tokens: 拼接后的词元列表
        - segments: 片段索引列表
    """
    tokens = ['<cls>'] + tokens_a + ['<sep>'] + tokens_b + ['<sep>']
    segments = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    return tokens, segments


def accuracy(y_hat, y):
    """计算预测正确的数量
    
    参数:
        y_hat: 预测结果，形状为 (batch_size, num_classes)
        y: 真实标签，形状为 (batch_size,)
    
    返回:
        预测正确的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
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


## 15.7.1 加载预训练的BERT
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout, max_len, devices):
    """加载预训练的BERT模型
    
    参数:
        pretrained_model: 预训练模型名称 ('bert.base' 或 'bert.small')
        num_hiddens: 隐藏层维度
        ffn_num_hiddens: 前馈网络隐藏层维度
        num_heads: 注意力头数
        num_layers: 编码器层数
        dropout: dropout率
        max_len: 最大序列长度
        devices: 设备列表
    
    返回:
        (bert, vocab) 元组
        - bert: BERT模型
        - vocab: 词表对象
    """
    if pretrained_model == 'bert.base':
        from downloader.bert_base import download_extract
        data_dir = download_extract()
    elif pretrained_model == 'bert.small':
        from downloader.bert_small import download_extract
        data_dir = download_extract()
    else:
        raise ValueError(f"未知的预训练模型: {pretrained_model}")
    
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    
    bert = BERTModel(len(vocab), num_hiddens, norm_shape=[num_hiddens],
                        ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens, 
                        num_heads=4, num_layers=2, dropout=0.2, 
                        max_len=max_len, key_size=256, query_size=256,
                        value_size=256, hid_in_features=256,
                        mlm_in_features=256, nsp_in_features=256)
    bert.load_state_dict(torch.load(os.path.join(data_dir, 'pretrained.params')))
    return bert, vocab


devices = try_all_gpus()
bert, vocab = load_pretrained_model('bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4, num_layers=2, dropout=0.1, max_len=512, devices=devices)


## 15.7.2 微调BERT的数据集
class SNLIBERTDataset(torch.utils.data.Dataset):
    """SNLI BERT数据集类
    
    继承自PyTorch的Dataset类，用于加载和预处理SNLI数据集以供BERT使用。
    
    参数:
        dataset: 原始数据集元组 (premises, hypotheses, labels)
        max_len: 最大序列长度
        vocab: 词表对象
    """
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[p_tokens, h_tokens]
                                         for p_tokens, h_tokens in zip(
                                             *[tokenize([s.lower() for s in sentences])
                                               for sentences in dataset[:2]])]

        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments, self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        """预处理所有样本
        
        参数:
            all_premise_hypothesis_tokens: 所有的前提和假设词元列表
        
        返回:
            (all_token_ids, all_segments, valid_lens) 元组
        """
        pool = multiprocessing.Pool(4)
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        """多进程工作函数
        
        参数:
            premise_hypothesis_tokens: 前提和假设的词元列表
        
        返回:
            (token_ids, segments, valid_len) 元组
        """
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        """截断词元对
        
        为BERT输入保留'<CLS>'、'<SEP>'和'<SEP>'的位置。
        
        参数:
            p_tokens: 前提词元列表
            h_tokens: 假设词元列表
        """
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        """获取指定索引的数据样本
        
        参数:
            idx: 样本索引
        
        返回:
            ((token_ids, segments, valid_lens), label) 元组
        """
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        """返回数据集大小"""
        return len(self.all_token_ids)


batch_size, max_len, num_workers = 512, 128, get_dataloader_workers()
data_dir = download_extract('SNLI')
train_set = SNLIBERTDataset(read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size, num_workers=num_workers)


## 15.7.3 微调BERT
class BERTClassifier(nn.Module):
    """BERT分类器
    
    使用预训练的BERT编码器进行自然语言推断任务的分类。
    
    参数:
        bert: 预训练的BERT模型
    """
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.linear = nn.Linear(256, 3)

    def forward(self, inputs):
        """前向传播
        
        参数:
            inputs: 输入元组 (tokens_X, segments_X, valid_lens_x)
        
        返回:
            分类结果，形状为 (batch_size, 3)
        """
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.linear(self.hidden(encoded_X[:, 0, :]))


net = BERTClassifier(bert)
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
