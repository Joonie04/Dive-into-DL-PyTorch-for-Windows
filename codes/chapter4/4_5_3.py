import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
from matplotlib import pyplot as plt  # 导入matplotlib.pyplot绘图模块

# 生成合成数据
def synthetic_data(w, b, num_examples):  # 定义生成合成数据的函数
    """生成y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 生成服从标准正态分布的特征X
    y = torch.matmul(X, w) + b  # 计算y = Xw + b
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声
    return X, y.reshape((-1, 1))  # 返回特征X和标签y

# 加载数据
def load_array(data_arrays, batch_size, is_train=True):  # 定义构造PyTorch数据迭代器的函数
    """构造一个PyTorch数据迭代器"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)  # 创建数据集
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)  # 创建数据迭代器

# 累加器类，用于累积损失和样本数量
class Accumulator:  # 定义累加器类
    def __init__(self, n):  # 初始化函数
        self.data = [0.0] * n  # 初始化数据列表
    
    def add(self, *args):  # 添加数据
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # 累加数据
    
    def reset(self):  # 重置数据
        self.data = [0.0] * len(self.data)  # 重置为0
    
    def __getitem__(self, idx):  # 获取数据项
        return self.data[idx]  # 返回指定索引的数据

# 评估损失
def evaluate_loss(net, data_iter, loss):  # 定义评估给定数据集上模型损失的函数
    """评估给定数据集上模型的损失"""
    metric = Accumulator(2)  # 损失的总和, 样本数量
    with torch.no_grad():  # 在不计算梯度的上下文中执行
        for X, y in data_iter:  # 遍历数据迭代器
            out = net(X)  # 前向传播
            y = y.reshape(out.shape)  # 重塑y的形状
            l = loss(out, y)  # 计算损失
            metric.add(l.sum(), l.numel())  # 累加损失和样本数
    return metric[0] / metric[1]  # 返回平均损失

# 绘制图表
def plot(X, Y, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):  # 定义绘制数据点的函数
    """绘制数据点"""
    if legend is None:  # 如果没有提供图例
        legend = []  # 初始化空列表
    
    set_figsize(figsize)  # 设置图表大小
    axes = axes if axes else plt.gca()  # 使用传入的axes或获取当前axes
    
    # 检查X和Y是否为列表或数组
    def has_one_axis(X):  # 定义检查X是否为一维的函数
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))  # 返回是否为一维
    
    if has_one_axis(X):  # 如果X是一维的
        X = [X] * len(Y)  # 为每个Y复制X
    
    axes.cla()  # 清除axes
    for x, y, fmt in zip(X, Y, fmts):  # 遍历X、Y和格式
        if len(x):  # 如果x非空
            axes.plot(x, y, fmt)  # 绘制曲线
        else:  # 如果x为空
            axes.plot(y, fmt)  # 只绘制y
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)  # 设置坐标轴

# 设置图表大小
def set_figsize(figsize=(3.5, 2.5)):  # 定义设置matplotlib图表大小的函数
    """设置matplotlib的图表大小"""
    plt.rcParams['figure.figsize'] = figsize  # 设置图表大小

# 设置图表坐标轴
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):  # 定义设置matplotlib坐标轴的函数
    """设置matplotlib的坐标轴"""
    axes.set_xlabel(xlabel)  # 设置x轴标签
    axes.set_ylabel(ylabel)  # 设置y轴标签
    axes.set_xscale(xscale)  # 设置x轴缩放
    axes.set_yscale(yscale)  # 设置y轴缩放
    axes.set_xlim(xlim)  # 设置x轴范围
    axes.set_ylim(ylim)  # 设置y轴范围
    if legend:  # 如果有图例
        axes.legend(legend)  # 显示图例
    axes.grid()  # 显示网格

# 定义全局变量
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5  # 设置训练样本数、测试样本数、输入维度和批次大小
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05  # 设置真实的权重和偏置
train_data = synthetic_data(true_w, true_b, n_train)  # 生成训练数据
train_iter = load_array(train_data, batch_size)  # 创建训练数据迭代器
test_data = synthetic_data(true_w, true_b, n_test)  # 生成测试数据
test_iter = load_array(test_data, batch_size, is_train=False)  # 创建测试数据迭代器

def init_params():  # 定义初始化参数的函数
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)  # 初始化权重
    b = torch.zeros(1, requires_grad=True)  # 初始化偏置
    return [w, b]  # 返回参数列表

def l2_penalty(w):  # 定义L2正则化惩罚项的函数
    return torch.sum(w.pow(2)) / 2  # 返回权重的L2范数的一半

def linreg(X, w, b):  # 定义线性回归模型的函数
    """线性回归模型"""
    return torch.matmul(X, w) + b  # 返回预测值

def squared_loss(y_hat, y):  # 定义均方误差损失函数
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # 返回均方误差

def train(lambd):  # 定义训练函数
    w, b = init_params()  # 初始化参数
    net, loss = lambda X: linreg(X, w, b), squared_loss  # 定义模型和损失函数
    num_epochs, lr = 100, 0.003  # 设置训练轮数和学习率
    
    # 记录训练和测试损失
    train_losses = []  # 初始化训练损失列表
    test_losses = []  # 初始化测试损失列表
    epochs = []  # 初始化epoch列表
    
    for epoch in range(num_epochs):  # 遍历每个epoch
        for X, y in train_iter:  # 遍历训练数据迭代器
            l = loss(net(X), y) + lambd * l2_penalty(w)  # 计算损失（包含L2正则化）
            l.sum().backward()  # 反向传播
            with torch.no_grad():  # 在不计算梯度的上下文中执行
                w -= lr * w.grad  # 更新权重
                b -= lr * b.grad  # 更新偏置
                w.grad.zero_()  # 清零权重梯度
                b.grad.zero_()  # 清零偏置梯度
        if (epoch + 1) % 5 == 0:  # 每5个epoch记录一次
            epochs.append(epoch + 1)  # 记录epoch
            train_loss = evaluate_loss(net, train_iter, loss)  # 评估训练损失
            test_loss = evaluate_loss(net, test_iter, loss)  # 评估测试损失
            train_losses.append(train_loss)  # 记录训练损失
            test_losses.append(test_loss)  # 记录测试损失
            print(f'epoch {epoch + 1}, train loss {train_loss:.6f}, test loss {test_loss:.6f}')  # 打印损失
    
    # 绘制损失曲线
    plot(epochs, [train_losses, test_losses], xlabel='epochs', ylabel='loss', yscale='log',
         xlim=[5, num_epochs], legend=['train', 'test'])  # 绘制训练和测试损失曲线
    plt.show()  # 显示图形
    
    print('w的L2范数是：', torch.norm(w).item())  # 打印权重的L2范数

train(lambd=0)  # 训练模型（不使用L2正则化）
train(lambd=3)  # 训练模型（使用L2正则化）
