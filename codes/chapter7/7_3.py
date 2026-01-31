# 7.3 网络中的网络(NiN)

import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
import torchvision  # 导入torchvision库用于数据集
from torchvision import transforms  # 导入数据变换模块
import sys  # 导入系统模块
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import time  # 导入时间模块用于计时


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


# 累加器类，用于在训练过程中累积多个指标
class Accumulator:  # 定义累加器类
    def __init__(self, n):  # 初始化函数，n表示要累加的指标数量
        self.data = [0.0] * n  # 创建长度为n的列表，初始化为0
    
    def add(self, *args):  # 添加数据的方法
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # 将新数据累加到对应位置
    
    def reset(self):  # 重置累加器的方法
        self.data = [0.0] * len(self.data)  # 将所有数据重置为0
    
    def __getitem__(self, idx):  # 索引方法，允许通过索引访问数据
        return self.data[idx]  # 返回指定索引的数据


# 计算准确率的函数
def accuracy(y_hat, y):  # 定义准确率计算函数
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 如果预测结果有多个类别
        y_hat = y_hat.argmax(axis=1)  # 取预测概率最大的类别作为预测结果
    cmp = y_hat.type(y.dtype) == y  # 比较预测结果和真实标签是否相等
    return float(cmp.type(torch.float32).sum())  # 返回预测正确的数量


# 计时器类，用于测量代码执行时间
class Timer:  # 定义计时器类
    def __init__(self):  # 初始化函数
        self.times = []  # 存储所有计时记录
        self.start()  # 启动计时器
    
    def start(self):  # 启动计时器的方法
        self.tik = time.time()  # 记录开始时间
    
    def stop(self):  # 停止计时器的方法
        self.times.append(time.time() - self.tik)  # 计算并记录时间差
        return self.times[-1]  # 返回最后一次计时结果
    
    def avg(self):  # 计算平均时间的方法
        return sum(self.times) / len(self.times)  # 返回所有计时的平均值
    
    def sum(self):  # 计算总时间的方法
        return sum(self.times)  # 返回所有计时的总和
    
    def cumsum(self):  # 计算累计时间的方法
        return np.array(self.times).cumsum().tolist()  # 返回累计时间列表


# 尝试使用GPU的函数
def try_gpu(i=0):  # 定义尝试使用GPU的函数
    if torch.cuda.device_count() >= i + 1:  # 如果存在第i个GPU
        return torch.device(f"cuda:{i}")  # 返回第i个GPU设备
    return torch.device("cpu")  # 否则返回CPU设备


# 动画绘图类，用于可视化训练过程
class Animator:  # 定义动画绘图类
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):  # 初始化函数
        if legend is None:  # 如果没有提供图例
            legend = []  # 创建空图例列表
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)  # 创建图形和坐标轴
        if nrows * ncols == 1:  # 如果只有一个子图
            self.axes = [self.axes]  # 将坐标轴转换为列表
        self.config_axes = lambda: self.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)  # 配置坐标轴的函数
        self.X, self.Y, self.fmts = None, None, fmts  # 初始化X、Y和格式
        self.config_axes()  # 配置坐标轴
    
    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):  # 设置坐标轴的方法
        ax.set_xlabel(xlabel)  # 设置x轴标签
        ax.set_ylabel(ylabel)  # 设置y轴标签
        ax.set_xscale(xscale)  # 设置x轴缩放
        ax.set_yscale(yscale)  # 设置y轴缩放
        ax.set_xlim(xlim)  # 设置x轴范围
        ax.set_ylim(ylim)  # 设置y轴范围
        if legend:  # 如果有图例
            ax.legend(legend)  # 添加图例
        ax.grid()  # 添加网格
    
    def add(self, x, y):  # 添加数据点的方法
        if not hasattr(y, "__len__"):  # 如果y不是可迭代对象
            y = [y]  # 将y转换为列表
        n = len(y)  # 获取y的长度
        if not hasattr(x, "__len__"):  # 如果x不是可迭代对象
            x = [x] * n  # 将x重复n次
        if not self.X:  # 如果X为空
            self.X = [[] for _ in range(n)]  # 创建n个空列表
        if not self.Y:  # 如果Y为空
            self.Y = [[] for _ in range(n)]  # 创建n个空列表
        for i, (a, b) in enumerate(zip(x, y)):  # 遍历x和y
            if a is not None and b is not None:  # 如果a和b都不为None
                self.X[i].append(a)  # 添加x值
                self.Y[i].append(b)  # 添加y值
        self.axes[0].cla()  # 清除坐标轴
        for x, y, fmt in zip(self.X, self.Y, self.fmts):  # 遍历所有数据
            self.axes[0].plot(x, y, fmt)  # 绘制曲线
        self.config_axes()  # 重新配置坐标轴
        plt.draw()  # 绘制图形
        plt.pause(0.001)  # 暂停一小段时间


# 7.3.1 NiN块
def nin_block(in_channels, out_channels, kernel_size, strides, padding):  # 定义NiN块的函数
    return nn.Sequential(  # 返回一个序列模块
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),  # 第一个卷积层：使用指定的卷积核大小、步幅和填充，后接ReLU激活函数
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),  # 第二个卷积层：1x1卷积核，用于通道降维，后接ReLU激活函数
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())  # 第三个卷积层：1x1卷积核，用于通道降维，后接ReLU激活函数

# 7.3.2 NiN模型
net = nn.Sequential(  # 创建NiN网络
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),  # 第一个NiN块：1个输入通道，96个输出通道，11x11卷积核，步幅4，无填充
    nn.MaxPool2d(kernel_size=3, stride=2),  # 第一个最大池化层：3x3窗口，步幅2
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),  # 第二个NiN块：96个输入通道，256个输出通道，5x5卷积核，步幅1，填充2
    nn.MaxPool2d(kernel_size=3, stride=2),  # 第二个最大池化层：3x3窗口，步幅2
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),  # 第三个NiN块：256个输入通道，384个输出通道，3x3卷积核，步幅1，填充1
    nn.MaxPool2d(kernel_size=3, stride=2),  # 第三个最大池化层：3x3窗口，步幅2
    nn.Dropout(0.5),  # Dropout层：丢弃率为0.5，随机丢弃50%的神经元
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),  # 第四个NiN块：384个输入通道，10个输出通道（对应10个类别），3x3卷积核，步幅1，填充1
    nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化层：将特征图池化为1x1大小
    nn.Flatten())  # 展平层：将多维张量展平为一维

X = torch.rand(size=(1, 1, 224, 224))  # 创建一个随机输入张量，形状为（批量大小，通道数，高度，宽度）
for layer in net:  # 遍历网络中的每一层
    X = layer(X)  # 将输入通过当前层
    print(layer.__class__.__name__, 'output shape:\t', X.shape)  # 打印当前层的名称和输出形状


# 7.3.3 模型训练
lr, num_epochs, batch_size = 0.01, 10, 128  # 设置学习率为0.01，训练轮次为10，批量大小为128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)  # 加载FashionMNIST训练集和测试集，调整图像大小为224x224

def evaluate_accuracy_gpu(net, data_iter, device=None):  # 定义在GPU上评估模型准确率的函数
    if isinstance(net, nn.Module):  # 如果网络是PyTorch模块
        net.eval()  # 将网络设置为评估模式
        if not device:  # 如果没有指定设备
            device = next(iter(net.parameters())).device  # 获取网络参数所在的设备
    metric = Accumulator(2)  # 创建累加器，用于累加正确预测数和样本总数
    with torch.no_grad():  # 禁用梯度计算
        for X, y in data_iter:  # 遍历数据迭代器
            if isinstance(X, list):  # 如果X是列表
                X = [x.to(device) for x in X]  # 将列表中的每个张量移动到指定设备
            else:  # 如果X不是列表
                X = X.to(device)  # 将张量移动到指定设备
            y = y.to(device)  # 将标签移动到指定设备
            metric.add(accuracy(net(X), y), y.numel())  # 累加正确预测数和样本总数
    return metric[0] / metric[1]  # 返回准确率（正确预测数/样本总数）

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):  # 定义训练函数
    def init_weights(m):  # 定义权重初始化函数
        if type(m) == nn.Linear or type(m) == nn.Conv2d:  # 如果是全连接层或卷积层
            nn.init.xavier_uniform_(m.weight)  # 使用Xavier均匀分布初始化权重
    net.apply(init_weights)  # 对网络应用权重初始化
    print("training on", device)  # 打印训练设备
    net.to(device)  # 将网络移动到指定设备
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 创建SGD优化器，学习率为lr
    loss = nn.CrossEntropyLoss()  # 创建交叉熵损失函数
    animator = Animator(xlabel="epoch", xlim=[1, num_epochs], legend=["train loss", "train acc", "test acc"])  # 创建动画绘图对象
    timer = Timer()  # 创建计时器
    for epoch in range(num_epochs):  # 遍历所有训练轮次
        net.train()  # 将网络设置为训练模式
        metric = Accumulator(3)  # 创建累加器，用于累加训练损失、训练准确率和样本数
        for i, (X, y) in enumerate(train_iter):  # 遍历训练数据迭代器
            timer.start()  # 开始计时
            optimizer.zero_grad()  # 清零梯度
            X, y = X.to(device), y.to(device)  # 将数据和标签移动到指定设备
            y_hat = net(X)  # 前向传播，计算预测结果
            l = loss(y_hat, y)  # 计算损失
            l.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数
            with torch.no_grad():  # 禁用梯度计算
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])  # 累加训练损失、训练准确率和样本数
            timer.stop()  # 停止计时
            train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]  # 计算平均训练损失和训练准确率
            if (i + 1) % 50 == 0:  # 每50个批次更新一次动画
                animator.add(epoch + (i + 1) / len(train_iter), (train_loss, train_acc, None))  # 添加训练损失和训练准确率到动画
        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 在测试集上评估模型
        animator.add(epoch + 1, (None, None, test_acc))  # 添加测试准确率到动画
    print(f"loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")  # 打印最终训练损失、训练准确率和测试准确率
    print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}")  # 打印训练速度

train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())  # 开始训练NiN模型
