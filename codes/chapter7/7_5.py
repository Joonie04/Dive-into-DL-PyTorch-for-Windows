# 7.5 批量规范化

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


# 7.5.3 从零实现
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps=1e-5, momentum=0.9):  # 定义批量规范化函数
    # 通过`is_grad_enabled`来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():  # 如果是在预测模式下
        # 直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)  # 使用移动均值和方差进行标准化
    else:  # 如果是在训练模式下
        assert len(X.shape) in (2, 4)  # 断言输入形状是2维或4维
        if len(X.shape) == 2:  # 使用全连接层的情况
            # 计算特征维上的均值和方差
            mean = X.mean(dim=0)  # 计算均值
            var = ((X - mean) ** 2).mean(dim=0)  # 计算方差
        else:  # 使用二维卷积层的情况
            # 计算通道维上（axis=1）的均值和方差
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)  # 计算均值，保持维度
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)  # 计算方差，保持维度
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)  # 使用当前均值和方差进行标准化
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean  # 更新移动均值
        moving_var = momentum * moving_var + (1.0 - momentum) * var  # 更新移动方差
    Y = gamma * X_hat + beta  # 应用缩放和偏移
    return Y, moving_mean.data, moving_var.data  # 返回输出、更新后的移动均值和移动方差

class BatchNorm(nn.Module):  # 定义批量规范化层类
    # num_features：完全连接层的输出数量或卷积层的输出通道数
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        if num_dims == 2:  # 如果是全连接层
            shape = (1, num_features)  # 形状为（1，特征数）
        else:  # 如果是卷积层
            shape = (1, num_features, 1, 1)  # 形状为（1，特征数，1，1）
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))  # 拉伸参数gamma，初始化为1
        self.beta = nn.Parameter(torch.zeros(shape))  # 偏移参数beta，初始化为0
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)  # 移动均值，初始化为0
        self.moving_var = torch.ones(shape)  # 移动方差，初始化为1
        self.eps = 1e-5  # 用于数值稳定的小常数
        self.momentum = 0.9  # 用于更新移动平均的动量
    
    def forward(self, X):  # 前向传播函数
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:  # 如果设备不一致
            self.moving_mean = self.moving_mean.to(X.device)  # 将移动均值移动到X的设备
            self.moving_var = self.moving_var.to(X.device)  # 将移动方差移动到X的设备
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(  # 调用批量规范化函数
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,  # 传入参数
            eps=self.eps, momentum=self.momentum)  # 传入eps和momentum
        return Y  # 返回规范化后的输出


# 7.5.4 使用批量规范化层的LeNet
net = nn.Sequential(  # 创建使用批量规范化层的LeNet网络
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),  # 第一个卷积层：1→6通道，5x5卷积核，批量规范化，Sigmoid激活
    nn.AvgPool2d(kernel_size=2, stride=2),  # 第一个平均池化层：2x2窗口，步幅2
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),  # 第二个卷积层：6→16通道，5x5卷积核，批量规范化，Sigmoid激活
    nn.AvgPool2d(kernel_size=2, stride=2),  # 第二个平均池化层：2x2窗口，步幅2
    nn.Flatten(),  # 展平层：将多维张量展平为一维
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),  # 第一个全连接层：256→120，批量规范化，Sigmoid激活
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),  # 第二个全连接层：120→84，批量规范化，Sigmoid激活
    nn.Linear(84, 10))  # 输出层：84→10（10个类别）

lr, num_epochs, batch_size = 1.0, 10, 256  # 设置学习率为1.0，训练轮次为10，批量大小为256
train_iter, test_iter = load_data_fashion_mnist(batch_size)  # 加载FashionMNIST训练集和测试集

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

train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())  # 开始训练模型

print("net[1].gamma.reshape((-1,))", net[1].gamma.reshape((-1,)))  # 打印第一个批量规范化层的gamma参数
print("net[1].beta.reshape((-1,))", net[1].beta.reshape((-1,)))  # 打印第一个批量规范化层的beta参数


# 7.5.5 简洁实现
net = nn.Sequential(  # 创建使用PyTorch内置批量规范化层的LeNet网络
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),  # 第一个卷积层：1→6通道，5x5卷积核，批量规范化，Sigmoid激活
    nn.AvgPool2d(kernel_size=2, stride=2),  # 第一个平均池化层：2x2窗口，步幅2
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),  # 第二个卷积层：6→16通道，5x5卷积核，批量规范化，Sigmoid激活
    nn.AvgPool2d(kernel_size=2, stride=2),  # 第二个平均池化层：2x2窗口，步幅2
    nn.Flatten(),  # 展平层：将多维张量展平为一维
    nn.Linear(16*4*4, 120), nn.BatchNorm1d(120), nn.Sigmoid(),  # 第一个全连接层：256→120，批量规范化，Sigmoid激活
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),  # 第二个全连接层：120→84，批量规范化，Sigmoid激活
    nn.Linear(84, 10))  # 输出层：84→10（10个类别）

train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())  # 开始训练模型
