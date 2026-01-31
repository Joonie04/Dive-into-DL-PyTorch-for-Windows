# 13.13 实战Kaggle竞赛: 图像分类(CIFAR-10)
import collections
import math
import os
import shutil
import pandas as pd
import time
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

## 13.13.1 获取并组织数据集

### 1. 下载数据集
demo = True

if demo:
    from downloader.cifar10_tiny import download_cifar10_tiny
    data_dir = download_cifar10_tiny()
    if data_dir is None:
        raise RuntimeError("数据集下载失败")
else:
    data_dir = '../data/cifar-10'

### 2. 整理数据集
def read_csv_labels(fname):
    """读取 CSV 标签文件
    
    参数:
        fname: CSV 文件路径
    
    返回:
        字典，键为文件名，值为标签
    """
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print("# 训练样本:", len(labels))
print("# 类别:", len(set(labels.values())))

def copyfile(filename, target_dir):
    """将文件复制到目标目录
    
    参数:
        filename: 源文件路径
        target_dir: 目标目录路径
    """
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来
    
    参数:
        data_dir: 数据集目录
        labels: 标签字典
        valid_ratio: 验证集比例
    
    返回:
        每个类别的验证集样本数
    """
    n = collections.Counter(labels.values()).most_common()[-1][1]
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))

    return n_valid_per_label

def reorg_test(data_dir):
    """在预测期间组织测试集
    
    参数:
        data_dir: 数据集目录
    """
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file), os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))

def reorg_cifar10(data_dir, valid_ratio):
    """整理 CIFAR-10 数据集
    
    参数:
        data_dir: 数据集目录
        valid_ratio: 验证集比例
    """
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10(data_dir, valid_ratio)


## 13.13.2 图像增广
trainsform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


## 13.13.3 读取数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=trainsform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)


## 13.13.4 定义模型
class Residual(nn.Module):
    """残差块
    
    残差块是 ResNet 的基本构建块，通过跳跃连接解决了深度网络中的梯度消失问题。
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        use_1x1conv: 是否使用 1x1 卷积调整维度
        stride: 卷积步长
    """
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        """前向传播
        
        参数:
            X: 输入张量
        
        返回:
            输出张量
        """
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet18(num_classes, in_channels=3):
    """构建 ResNet-18 模型
    
    ResNet-18 是一个 18 层的残差网络，包含多个残差块。
    
    参数:
        num_classes: 类别数
        in_channels: 输入通道数
    
    返回:
        ResNet-18 模型
    """
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        """构建残差块
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_residuals: 残差块数量
            first_block: 是否为第一个块
        
        返回:
            残差块序列
        """
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 64, 2))
    net.add_module("resnet_block3", resnet_block(64, 128, 2))
    net.add_module("resnet_block4", resnet_block(128, 128, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes)))
    return net

def get_net(device):
    """获取模型
    
    参数:
        device: 计算设备
    
    返回:
        ResNet-18 模型
    """
    num_classes = 10
    net = resnet18(num_classes, 3)
    return net.to(device)

loss = nn.CrossEntropyLoss(reduction='none')


## 13.13.5 定义训练函数
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

def accuracy(y_hat, y):
    """计算预测准确率
    
    参数:
        y_hat: 预测值
        y: 真实值
    
    返回:
        准确率
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(torch.float32).sum())

def try_all_gpus():
    """尝试获取所有可用的 GPU
    
    返回:
        GPU 设备列表，如果没有 GPU 则返回 CPU 设备列表
    """
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(torch.device(f'cuda:{i}'))
    return devices if devices else [torch.device('cpu')]

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

def train_batch_ch13(net, X, y, loss, trainer, devices):
    """训练一个批次的数据
    
    参数:
        net: 模型
        X: 输入数据
        y: 标签
        loss: 损失函数
        trainer: 优化器
        devices: 设备列表
    
    返回:
        训练损失和训练准确率
    """
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """在 GPU 上评估模型的准确率
    
    参数:
        net: 模型
        data_iter: 数据迭代器
        device: 计算设备
    
    返回:
        准确率
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    """用多 GPU 训练模型
    
    参数:
        net: 模型
        train_iter: 训练数据迭代器
        valid_iter: 验证数据迭代器
        num_epochs: 训练轮数
        lr: 学习率
        wd: 权重衰减
        devices: 设备列表
        lr_period: 学习率衰减周期
        lr_decay: 学习率衰减率
    """
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    net.train()
    metric = Accumulator(3)
    valid_acc = 0
    for epoch in range(num_epochs):
        metric.reset()
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[2], None))
        if valid_iter is not None:
            valid_acc = evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')


## 13.13.6 训练和验证模型
devices, num_epochs, lr, wd, lr_period, lr_decay, net = try_all_gpus(), 10, 2e-2, 5e-4, 4, 0.9, get_net(devices[0])
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)


## 13.13.7 在 Kaggle 上对测试集进行分类并提交结果
net, preds = get_net(devices[0]), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)
for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    print("=" * 60)
    print("Kaggle 竞赛: CIFAR-10 图像分类")
    print("=" * 60)
    
    print("\n数据集信息:")
    print(f"  - 训练样本数: {len(train_ds)}")
    print(f"  - 验证样本数: {len(valid_ds)}")
    print(f"  - 测试样本数: {len(test_ds)}")
    print(f"  - 类别数: {len(train_ds.classes)}")
    print(f"  - 类别: {train_ds.classes}")
    
    print("\n模型信息:")
    print(f"  - 模型: ResNet-18")
    print(f"  - 输入通道数: 3")
    print(f"  - 输出类别数: 10")
    
    print("\n训练参数:")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 训练轮数: {num_epochs}")
    print(f"  - 学习率: {lr}")
    print(f"  - 权重衰减: {wd}")
    print(f"  - 学习率衰减周期: {lr_period}")
    print(f"  - 学习率衰减率: {lr_decay}")
    print(f"  - 设备: {devices}")
