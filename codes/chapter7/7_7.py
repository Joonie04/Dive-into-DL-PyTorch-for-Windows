# 7.7 稠密连接网络(DenseNet)

# 7.7.2 稠密块体
import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
from torch.nn import functional as F  # 导入PyTorch函数式接口
import torchvision  # 导入torchvision库，用于计算机视觉任务
import torchvision.transforms as transforms  # 导入图像变换模块
from matplotlib import pyplot as plt  # 导入matplotlib.pyplot绘图模块
import sys  # 导入系统模块
import os  # 导入操作系统模块
import time  # 导入时间模块
import numpy as np  # 导入numpy库

# 设置数据集路径
dataset_path = 'dataset/FashionMNIST/raw'  # 设置FashionMNIST数据集存储路径

# 加载FashionMNIST数据集
def load_data_fashion_mnist(batch_size, resize=None):  # 定义加载FashionMNIST数据集的函数
    trans = []  # 初始化变换列表
    if resize:  # 如果需要调整大小
        trans.append(transforms.Resize(size=resize))  # 添加调整大小的变换
    trans.append(transforms.ToTensor())  # 添加转换为张量的变换
    
    transform = transforms.Compose(trans)  # 组合所有变换
    mnist_train = torchvision.datasets.FashionMNIST(root=dataset_path, train=True, download=False, transform=transform)  # 加载训练集
    mnist_test = torchvision.datasets.FashionMNIST(root=dataset_path, train=False, download=False, transform=transform)  # 加载测试集
    
    if sys.platform.startswith('win'):  # 如果是Windows系统
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:  # 如果是其他系统
        num_workers = 4  # 使用4个工作进程
    
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # 创建训练数据迭代器
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # 创建测试数据迭代器
    
    return train_iter, test_iter  # 返回训练和测试数据迭代器

# 累加器类，用于累加多个变量
class Accumulator:  # 定义累加器类
    def __init__(self, n):  # 初始化函数，n为需要累加的变量数量
        self.data = [0.0] * n  # 初始化n个累加变量为0.0
    
    def add(self, *args):  # 添加函数，用于累加多个值
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # 将每个累加变量加上对应的值
    
    def reset(self):  # 重置函数
        self.data = [0.0] * len(self.data)  # 将所有累加变量重置为0.0
    
    def __getitem__(self, idx):  # 索引函数，用于获取指定索引的累加变量
        return self.data[idx]  # 返回指定索引的累加变量

# 计算准确率
def accuracy(y_hat, y):  # 定义计算准确率的函数
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 如果预测结果是多维的（即包含多个类别的概率）
        y_hat = y_hat.argmax(axis=1)  # 取每行最大值的索引作为预测类别
    cmp = y_hat.type(y.dtype) == y  # 比较预测类别和真实类别是否相等
    return float(cmp.type(torch.float32).sum())  # 返回正确预测的数量

# 评估模型准确率
def evaluate_accuracy_gpu(net, data_iter, device=None):  # 定义评估模型准确率的函数，使用GPU
    if isinstance(net, nn.Module):  # 如果是PyTorch模型
        net.eval()  # 设置为评估模式
        if not device:  # 如果没有指定设备
            device = next(iter(net.parameters())).device  # 获取模型所在的设备
    metric = Accumulator(2)  # 创建累加器，用于累加正确预测数和样本总数
    
    with torch.no_grad():  # 在不计算梯度的上下文中执行
        for X, y in data_iter:  # 遍历数据迭代器
            X = X.to(device)  # 将输入数据移动到指定设备
            y = y.to(device)  # 将标签移动到指定设备
            metric.add(accuracy(net(X), y), y.numel())  # 累加正确预测数和样本总数
    return metric[0] / metric[1]  # 返回准确率

# 计时器类
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
        return np.array(self.times).cumsum().tolist()  # 返回累积时间列表

# 动画绘制类
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

# 尝试使用GPU
def try_gpu(i=0):  # 定义尝试使用GPU的函数
    if torch.cuda.device_count() >= i + 1:  # 如果GPU数量大于等于i+1
        return torch.device(f'cuda:{i}')  # 返回第i个GPU设备
    else:  # 如果GPU数量不足
        return torch.device('cpu')  # 返回CPU设备

# 训练函数（使用GPU）
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):  # 定义训练模型的函数，使用GPU
    def init_weights(m):  # 定义初始化权重的函数
        if type(m) == nn.Linear or type(m) == nn.Conv2d:  # 如果是全连接层或卷积层
            nn.init.xavier_uniform_(m.weight)  # 使用Xavier均匀分布初始化权重
    
    net.apply(init_weights)  # 对网络应用初始化权重函数
    print('training on', device)  # 打印训练设备
    net.to(device)  # 将网络移动到指定设备
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 定义随机梯度下降优化器
    loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])  # 创建动画绘制对象
    timer, num_batches = Timer(), len(train_iter)  # 创建计时器并获取批次数量
    for epoch in range(num_epochs):  # 遍历每个epoch
        metric = Accumulator(3)  # 创建累加器，用于累加训练损失、训练准确率和样本总数
        net.train()  # 设置为训练模式
        for i, (X, y) in enumerate(train_iter):  # 遍历训练数据迭代器
            timer.start()  # 启动计时器
            optimizer.zero_grad()  # 清零梯度
            X, y = X.to(device), y.to(device)  # 将输入数据和标签移动到指定设备
            y_hat = net(X)  # 前向传播，计算预测值
            l = loss(y_hat, y)  # 计算损失
            l.backward()  # 反向传播计算梯度
            optimizer.step()  # 使用优化器更新参数
            with torch.no_grad():  # 在不计算梯度的上下文中执行
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])  # 累加训练损失、训练准确率和样本总数
            timer.stop()  # 停止计时器
            train_l = metric[0] / metric[2]  # 计算平均训练损失
            train_acc = metric[1] / metric[2]  # 计算训练准确率
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:  # 每处理5个批次或最后一个批次时更新动画
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))  # 更新动画
        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 评估测试集准确率
        animator.add(epoch + 1, (None, None, test_acc))  # 更新动画，添加测试准确率
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '  # 打印训练损失和训练准确率
          f'test acc {test_acc:.3f}')  # 打印测试准确率
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '  # 打印每秒处理的样本数
          f'on {str(device)}')  # 打印使用的设备

# 定义卷积块函数
def conv_block(input_channels, num_channels):  # 定义卷积块的函数
    return nn.Sequential(  # 返回一个序列模块
        nn.BatchNorm2d(input_channels), nn.ReLU(),  # 批量规范化层 + ReLU激活函数
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))  # 卷积层：3x3卷积核，填充为1

# 定义稠密块类
class DenseBlock(nn.Module):  # 定义稠密块类，继承自nn.Module
    def __init__(self, num_convs, input_channels, num_channels):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        layer = []  # 初始化层列表
        for i in range(num_convs):  # 循环创建指定数量的卷积块
            layer.append(conv_block(  # 添加卷积块
                num_channels * i + input_channels, num_channels))  # 输入通道数为num_channels * i + input_channels，输出通道数为num_channels
        self.net = nn.Sequential(*layer)  # 将层列表转换为序列模块

    def forward(self, X):  # 前向传播函数
        for blk in self.net:  # 遍历网络中的每个块
            Y = blk(X)  # 将输入X通过当前块
            X = torch.cat((X, Y), dim=1)  # 在通道维度上连接输入X和输出Y
        return X  # 返回连接后的结果

# 测试稠密块
blk = DenseBlock(2, 3, 10)  # 创建一个稠密块，包含2个卷积块，输入通道数为3，每个卷积块输出通道数为10
X = torch.randn(4, 3, 8, 8)  # 创建一个随机输入张量，形状为(4, 3, 8, 8)
Y = blk(X)  # 通过稠密块
print("Y.shape", Y.shape)  # 打印输出形状，应该是(4, 23, 8, 8)，因为3 + 10 + 10 = 23


# 7.7.3 过渡层
# 定义过渡层函数
def transition_block(input_channels, num_channels):  # 定义过渡层的函数
    return nn.Sequential(  # 返回一个序列模块
        nn.BatchNorm2d(input_channels), nn.ReLU(),  # 批量规范化层 + ReLU激活函数
        nn.Conv2d(input_channels, num_channels, kernel_size=1),  # 1x1卷积层：用于减少通道数
        nn.AvgPool2d(kernel_size=2, stride=2))  # 平均池化层：2x2窗口，步幅2，用于减小特征图尺寸

# 测试过渡层
blk = transition_block(23, 10)  # 创建一个过渡层，输入通道数为23，输出通道数为10
print("blk(Y).shape", blk(Y).shape)  # 打印输出形状，应该是(4, 10, 4, 4)，因为通道数从23减少到10，尺寸从8x8减少到4x4
print("torch.Size([4, 10, 4, 4])", torch.Size([4, 10, 4, 4]))  # 打印期望的输出形状


# 7.7.4 DenseNet模型
# 定义DenseNet的第一个块
b1 = nn.Sequential(  # 使用nn.Sequential创建第一个块
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 卷积层：1个输入通道（灰度图像），64个输出通道，7x7卷积核，步幅2，填充3
    nn.BatchNorm2d(64), nn.ReLU(),  # 批量规范化层 + ReLU激活函数
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 最大池化层：3x3窗口，步幅2，填充1

# 设置DenseNet的参数
num_channels, growth_rate = 64, 32  # 初始通道数为64，增长率为32（每个卷积块增加的通道数）
num_convs_in_dense_blocks = [4, 4, 4, 4]  # 每个稠密块中卷积块的数量

# 构建DenseNet的各个块
blks = []  # 初始化块列表
for i, num_convs in enumerate(num_convs_in_dense_blocks):  # 遍历每个稠密块
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))  # 添加稠密块
    num_channels += num_convs * growth_rate  # 更新通道数：加上所有卷积块增加的通道数
    if i != len(num_convs_in_dense_blocks) - 1:  # 如果不是最后一个稠密块
        blks.append(transition_block(num_channels, num_channels // 2))  # 添加过渡层，将通道数减半
        num_channels = num_channels // 2  # 更新通道数

# 定义完整的DenseNet网络
net = nn.Sequential(  # 使用nn.Sequential创建网络
    b1, *blks,  # 第一个块和所有稠密块及过渡层
    nn.BatchNorm2d(num_channels), nn.ReLU(),  # 批量规范化层 + ReLU激活函数
    nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化层：将特征图大小调整为1x1
    nn.Flatten(),  # 展平层：将多维张量展平为一维
    nn.Linear(num_channels, 10))  # 全连接层：输入num_channels，输出10（10个类别，因为使用Fashion-MNIST）


# 7.7.5 训练模型
lr, num_epochs, batch_size = 0.1, 10, 256  # 设置学习率、训练轮数和批量大小
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)  # 加载FashionMNIST数据集，调整图像大小为96x96
train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())  # 训练模型，使用GPU
