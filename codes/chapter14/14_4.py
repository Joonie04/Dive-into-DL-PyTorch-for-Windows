# 14.4. 预训练word2vec
import math
import time
import os
import random
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# 从 14_3.py 导入数据加载函数
import sys
sys.path.insert(0, os.path.dirname(__file__))
from chapter14_14_3 import load_data_ptb as load_data_ptb_14_3

def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """加载 PTB 数据集的包装函数
    
    参数:
        batch_size: 批次大小
        max_window_size: 最大上下文窗口大小
        num_noise_words: 每个上下文词对应的负样本数量
    
    返回:
        (data_iter, vocab) 元组
    """
    return load_data_ptb_14_3(batch_size, max_window_size, num_noise_words)

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = load_data_ptb(batch_size, max_window_size, num_noise_words)


## 14.4.1 跳元模型

### 1. 嵌入层
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f"Parameter embedding_weight shape: {embed.weight.shape}")
print(f"dtype={embed.weight.dtype}")

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("embed(x):", embed(x))

### 2. 定义前向传播
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    """跳元模型的前向传播
    
    跳元模型（Skip-gram）是一种词嵌入学习方法。
    对于给定的中心词，我们预测其上下文词。
    
    参数:
        center: 中心词索引，形状为 (batch_size, 1)
        contexts_and_negatives: 上下文词和负样本索引，形状为 (batch_size, max_len)
        embed_v: 中心词的嵌入层
        embed_u: 上下文词的嵌入层
    
    返回:
        预测结果，形状为 (batch_size, max_len)
    """
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

skip_gram_shape = skip_gram(torch.ones((2, 1), dtype=torch.long), torch.ones((2, 4), dtype=torch.long), embed, embed).shape
print("skip_gram_shape:", skip_gram_shape)


## 14.4.2 训练

### 1. 二元交叉熵损失
class SigmoidBCELoss(nn.Module):
    """带 Sigmoid 的二元交叉熵损失
    
    该损失函数用于训练词嵌入模型。
    对于每个中心词-上下文词对，我们计算二元交叉熵损失。
    
    参数:
        无
    """
    def __init__(self):
        """初始化损失函数"""
        super().__init__()

    def forward(self, inputs, target, mask=None):
        """前向传播
        
        参数:
            inputs: 模型输出（logits）
            target: 目标标签（0 或 1）
            mask: 掩码，用于忽略填充的位置
        
        返回:
            损失值
        """
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()

pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
print("loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1):", loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))

def sigmoid(x):
    """Sigmoid 函数
    
    参数:
        x: 输入值
    
    返回:
        sigmoid(x) = -log(1 / (1 + exp(-x)))
    """
    return -math.log(1 / (1 + math.exp(-x)))

print(f"{(sigmoid(1.1) + sigmoid(-2.2) + sigmoid(3.3) + sigmoid(-4.4)) / 4:.4f}")
print(f"{(sigmoid(-1.1) + sigmoid(-2.2)) / 2:.4f}")

### 2. 初始化模型参数
embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size)
)

### 3. 定义训练阶段代码
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
    """累加器类，用于累积多个数值"""
    def __init__(self, n):
        """初始化累加器
        
        参数:
            n: 累加的数值数量
        """
        self.data = [0.0] * n
    
    def add(self, *args):
        """累加数值
        
        参数:
            *args: 要累加的数值
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

def try_gpu():
    """尝试获取 GPU 设备
    
    返回:
        GPU 设备，如果没有 GPU 则返回 CPU 设备
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class Animator:
    """动画可视化类，用于绘制训练过程中的指标变化"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        """初始化动画可视化类
        
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

def train(net, data_iter, lr, num_epochs, device=try_gpu()):
    """训练词嵌入模型
    
    参数:
        net: 词嵌入模型
        data_iter: 数据迭代器
        lr: 学习率
        num_epochs: 训练轮数
        device: 计算设备
    """
    def init_weights(m):
        """初始化模型权重
        
        参数:
            m: 模型层
        """
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = Animator(xlabel="epoch", ylabel="loss", xlim=[1, num_epochs])
    metric = Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, contexts_negatives, mask, label = [
                data.to(device) for data in batch
            ]
            pred = skip_gram(center, contexts_negatives, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[1], ))

    print(f"loss {metric[0] / metric[1]:.3f}, "
          f"{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}")

lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)


## 14.4.3 应用词嵌入
def get_similar_tokens(query_token, k, embed):
    """查找与查询词最相似的词
    
    使用余弦相似度来衡量词向量之间的相似性。
    余弦相似度计算公式: cos(a, b) = (a · b) / (||a|| * ||b||)
    
    参数:
        query_token: 查询词
        k: 返回的最相似词的数量
        embed: 词嵌入层
    
    返回:
        无，直接打印最相似的词
    """
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似度
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输入词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])

if __name__ == "__main__":
    print("=" * 60)
    print("14.4 预训练 word2vec")
    print("=" * 60)
    
    print("\n模型信息:")
    print(f"  - 词表大小: {len(vocab)}")
    print(f"  - 嵌入维度: {embed_size}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 最大上下文窗口大小: {max_window_size}")
    print(f"  - 负样本数量: {num_noise_words}")
    
    print("\n训练参数:")
    print(f"  - 学习率: {lr}")
    print(f"  - 训练轮数: {num_epochs}")
    print(f"  - 优化器: Adam")
    print(f"  - 损失函数: SigmoidBCELoss")
