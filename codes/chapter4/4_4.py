import math  # 导入数学模块
import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
from matplotlib import pyplot as plt  # 导入matplotlib.pyplot绘图模块

## 数据生成

max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配足够的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])  # 只使用前4个系数

features = np.random.normal(size=(n_train + n_test, 1))  # 生成随机特征
np.random.shuffle(features)  # 打乱特征顺序
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))  # 计算多项式特征
for i in range(max_degree):  # 遍历每个阶数
    poly_features[:, i] /= math.gamma(i + 1)  # 除以gamma函数（即i!）
# labels的维度为(n_train+n_test,)
labels = np.dot(poly_features, true_w)  # 计算标签
labels += np.random.normal(scale=0.1, size=labels.shape)  # 添加噪声

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]  # 转换为PyTorch张量

print(features[:2])  # 打印前2个特征
print(poly_features[:2])  # 打印前2个多项式特征
print(labels[:2])  # 打印前2个标签

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

# 加载数据
def load_array(data_arrays, batch_size, is_train=True):  # 定义构造PyTorch数据迭代器的函数
    """构造一个PyTorch数据迭代器"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)  # 创建数据集
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)  # 创建数据迭代器

# 训练epoch
def train_epoch_ch3(net, train_iter, loss, updater):  # 定义训练模型一个迭代周期的函数
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):  # 如果是PyTorch模型
        net.train()  # 设置为训练模式
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)  # 创建累加器
    for X, y in train_iter:  # 遍历训练数据迭代器
        # 计算梯度并更新参数
        y_hat = net(X)  # 前向传播
        l = loss(y_hat, y)  # 计算损失
        if isinstance(updater, torch.optim.Optimizer):  # 如果使用PyTorch优化器
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()  # 清零梯度
            l.mean().backward()  # 反向传播
            updater.step()  # 更新参数
        else:  # 如果使用自定义优化器
            # 使用定制的优化器和损失函数
            l.sum().backward()  # 反向传播
            updater(X.shape[0])  # 更新参数
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())  # 累加损失、准确率和样本数
    # 返回训练损失和训练准确度
    return metric[0] / metric[2], metric[1] / metric[2]  # 返回平均损失和准确率

# 评估准确率
def accuracy(y_hat, y):  # 定义计算预测正确数量的函数
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 如果是多分类
        y_hat = y_hat.argmax(axis=1)  # 获取预测类别
    cmp = y_hat.type(y.dtype) == y  # 比较预测和真实标签
    return float(cmp.type(y.dtype).sum())  # 返回正确预测数

# 评估损失
def evaluate_loss(net, data_iter, loss):  # 定义评估给定数据集上模型损失的函数
    """评估给定数据集上模型的损失"""
    metric = Accumulator(2)  # 损失的总和, 样本数量
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

# 训练函数
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):  # 定义训练函数
    loss = nn.MSELoss(reduction='none')  # 定义均方误差损失函数
    input_shape = train_features.shape[-1]  # 获取输入特征维度
    # 不设置偏置，因为我们已经在多项式特征中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))  # 定义线性模型（无偏置）
    batch_size = min(10, train_labels.shape[0])  # 设置批次大小
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)), batch_size)  # 创建训练数据迭代器
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)  # 创建测试数据迭代器
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)  # 定义随机梯度下降优化器
    
    # 记录训练和测试损失
    train_losses = []  # 初始化训练损失列表
    test_losses = []  # 初始化测试损失列表
    epochs = []  # 初始化epoch列表
    
    for epoch in range(num_epochs):  # 遍历每个epoch
        train_epoch_ch3(net, train_iter, loss, trainer)  # 训练一个epoch
        if epoch == 0 or (epoch + 1) % 20 == 0:  # 每20个epoch记录一次
            epochs.append(epoch + 1)  # 记录epoch
            train_loss = evaluate_loss(net, train_iter, loss)  # 评估训练损失
            test_loss = evaluate_loss(net, test_iter, loss)  # 评估测试损失
            train_losses.append(train_loss)  # 记录训练损失
            test_losses.append(test_loss)  # 记录测试损失
            print(f'epoch {epoch + 1}, train loss {train_loss:.6f}, test loss {test_loss:.6f}')  # 打印损失
    
    # 绘制损失曲线
    plot(epochs, [train_losses, test_losses], xlabel='epoch', ylabel='loss', yscale='log',
         xlim=[1, num_epochs], ylim=[1e-3, 1e2], legend=['train', 'test'])  # 绘制训练和测试损失曲线
    plt.show()  # 显示图形
    
    print('weight:', net[0].weight.data.numpy())  # 打印权重
    return net  # 返回模型

# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
net = train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])  # 训练模型（使用前4个特征）
print(net[0].weight.data.numpy())  # 打印权重

# 从多项式特征中选择前两个维度, 1,x
net = train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])  # 训练模型（使用前2个特征）
print(net[0].weight.data.numpy())  # 打印权重

# 从多项式特征中选择所有维度
net = train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)  # 训练模型（使用所有特征）
print(net[0].weight.data.numpy())  # 打印权重
