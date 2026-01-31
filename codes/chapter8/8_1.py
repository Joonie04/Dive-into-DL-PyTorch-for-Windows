# 8.1.2 训练

import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
from matplotlib import pyplot as plt  # 导入matplotlib.pyplot绘图模块
import numpy as np  # 导入numpy库

# 定义绘图函数
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):  # 定义绘图函数
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

# 定义加载数组的函数
def load_array(data_arrays, batch_size, is_train=True):  # 定义加载数组的函数
    dataset = torch.utils.data.TensorDataset(*data_arrays)  # 创建数据集
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)  # 创建数据迭代器

# 定义评估损失的函数
def evaluate_loss(net, data_iter, loss):  # 定义评估损失的函数
    metric = 0.0  # 初始化损失累加器
    n = 0  # 初始化样本数计数器
    for X, y in data_iter:  # 遍历数据迭代器
        l = loss(net(X), y)  # 计算损失
        metric += l.sum()  # 累加损失
        n += y.numel()  # 累加样本数
    return metric / n  # 返回平均损失

# 生成时间序列数据
T = 1000  # 设置时间步数
time = torch.arange(1, T + 1, dtype=torch.float32)  # 生成时间序列，从1到1000
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))  # 生成带有噪声的正弦波数据：sin(0.01 * t) + 噪声
plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6.3))  # 绘制时间序列数据

# 构建特征和标签
tau = 4  # 设置时间步长（使用前4个时间步预测下一个时间步）
features = torch.zeros((T - tau, tau))  # 初始化特征矩阵，形状为(T-tau, tau)
for i in range(tau):  # 遍历每个时间步
    features[:, i] = x[i: T - tau + i]  # 填充特征矩阵：第i列包含从第i个时间步开始的T-tau个值
labels = x[tau:].reshape((-1, 1))  # 标签为从第tau+1个时间步开始的值，形状为(T-tau, 1)

# 准备训练数据
batch_size, n_train = 16, 600  # 设置批量大小为16，训练样本数为600
train_iter = load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)  # 创建训练数据迭代器

# 定义权重初始化函数
def init_weights(m):  # 定义权重初始化函数
    if type(m) == nn.Linear:  # 如果是全连接层
        nn.init.xavier_uniform_(m.weight)  # 使用Xavier均匀分布初始化权重

# 定义网络构建函数
def get_net():  # 定义网络构建函数
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))  # 创建网络：4→10的全连接层 + ReLU + 10→1的全连接层
    net.apply(init_weights)  # 应用权重初始化
    return net  # 返回网络

# 定义损失函数
loss = nn.MSELoss(reduction='none')  # 定义均方误差损失函数，不进行降维

# 定义训练函数
def train(net, train_iter, loss, epochs, lr):  # 定义训练函数
    trainer = torch.optim.Adam(net.parameters(), lr)  # 定义Adam优化器
    for epoch in range(epochs):  # 遍历每个epoch
        for X, y in train_iter:  # 遍历训练数据迭代器
            trainer.zero_grad()  # 清零梯度
            l = loss(net(X), y)  # 计算损失
            l.sum().backward()  # 反向传播计算梯度
            trainer.step()  # 使用优化器更新参数
        print(f'epoch {epoch + 1}, '  # 打印epoch编号
              f'loss: {evaluate_loss(net, train_iter, loss):f}')  # 打印训练损失

# 训练网络
net = get_net()  # 获取网络
train(net, train_iter, loss, 5, 0.01)  # 训练网络，训练5个epoch，学习率为0.01

# 8.1.3 预测
# 单步预测
onest_step_preds = net(features)  # 使用训练好的网络进行单步预测
plot([time, time[tau:]], [x.detach().numpy(), onest_step_preds.detach().numpy()], 'time', 'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6.3))  # 绘制真实数据和单步预测结果

# 多步预测
multistep_preds = torch.zeros(T)  # 初始化多步预测结果
multistep_preds[: n_train + tau] = x[: n_train + tau]  # 前600+4个时间步使用真实数据
for i in range(n_train + tau, T):  # 从第604个时间步开始
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))  # 使用前4个预测值预测下一个值
plot([time, time[tau:], time[n_train + tau:]], [x.detach().numpy(), onest_step_preds.detach().numpy(), multistep_preds[n_train + tau:].detach().numpy()], 'time', 'x', legend=['data', '1-step preds', 'multistep preds'], xlim=[1, 1000], figsize=(6.3))  # 绘制真实数据、单步预测和多步预测结果

# 不同步长的预测
max_steps = 64  # 设置最大预测步数
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))  # 初始化特征矩阵，形状为(T-tau-max_steps+1, tau+max_steps)
for i in range(tau):  # 遍历前tau个时间步
    features[:, i] = x[i: T - tau - max_steps + 1 + i]  # 填充特征矩阵：使用真实数据
for i in range(tau, tau + max_steps):  # 遍历后max_steps个时间步
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)  # 使用前tau个预测值预测当前值

# 绘制不同步长的预测结果
steps = (1, 4, 16, 64)  # 设置不同的预测步数
plot([time[tau + i - 1: T - max_steps + i] for i in steps], [multistep_preds[tau + i - 1: T - max_steps + i].detach().numpy() for i in steps], 'time', 'x', legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000], figsize=(6.3))  # 绘制1步、4步、16步和64步预测结果
