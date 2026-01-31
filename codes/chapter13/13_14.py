# 13.14 实战Kaggle竞赛: 狗的品种识别(ImageNet Dogs)
import collections
import math
import os
import shutil
import time
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

## 13.14.1 下载和整理数据集

### 1. 下载数据集
demo = True

if demo:
    from downloader.dog_tiny import download_dog_tiny
    data_dir = download_dog_tiny()
    if data_dir is None:
        raise RuntimeError("数据集下载失败")
else:
    data_dir = '../data/dog-breed-identification'

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

def reorg_dog_data(data_dir, valid_ratio):
    """整理狗的品种识别数据集
    
    参数:
        data_dir: 数据集目录
        valid_ratio: 验证集比例
    """
    labels = read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)


## 13.14.2 图像增广
transform_train = torchvision.transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08到1之间，高宽比在3/4和4/3之间。
    # 然后，缩放图像以创建224x224的新图像
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # 随机更改图像的亮度，对比度和饱和度
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # 添加随机噪声
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])

transform_test = torchvision.transforms.Compose([
    # 缩放图像，使其最短边为224像素
    torchvision.transforms.Resize(256),
    # 从图像中心裁切224x224像素的图像
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])


## 13.14.3 读取数据集
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train_valid', 'train']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)


## 13.14.4 微调预训练模型
def get_net(device):
    """获取微调的 ResNet-34 模型
    
    使用预训练的 ResNet-34 作为特征提取器，并添加新的输出层用于狗的品种分类。
    
    参数:
        device: 计算设备
    
    返回:
        微调后的模型
    """
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 定义一个新的输出层，输出类别数为狗的品种数
    finetune_net.output_new = nn.Sequential(
        nn.Linear(1000, 256),
        nn.ReLU(),
        nn.Linear(256, 120))
    # 将模型参数移动到指定的设备上
    finetune_net = finetune_net.to(device)
    # 冻结特征提取层的参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, device):
    """评估模型的损失
    
    参数:
        data_iter: 数据迭代器
        net: 模型
        device: 计算设备
    
    返回:
        平均损失
    """
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(device), labels.to(device[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')


## 13.14.5 定义训练函数
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
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    valid_loss = 0
    for epoch in range(num_epochs):
        metric = Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().numpy()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')


## 13.14.6 训练和验证模型
devices, num_epochs, lr, wd = try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices[0])
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)


## 13.14.7 对测试集分类并在Kaggle提交结果
net = get_net(devices[0])
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)
preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for id, pred in zip(ids, preds):
        f.write(id.split('.')[0] + ',' + ','.join([f'{num:.6f}' for num in pred]) + '\n')

if __name__ == "__main__":
    print("=" * 60)
    print("Kaggle 竞赛: 狗的品种识别 (ImageNet Dogs)")
    print("=" * 60)
    
    print("\n数据集信息:")
    print(f"  - 训练样本数: {len(train_ds)}")
    print(f"  - 验证样本数: {len(valid_ds)}")
    print(f"  - 测试样本数: {len(test_ds)}")
    print(f"  - 类别数: {len(train_ds.classes)}")
    print(f"  - 类别数 (狗的品种): 120")
    
    print("\n模型信息:")
    print(f"  - 模型: ResNet-34 (预训练)")
    print(f"  - 输入通道数: 3")
    print(f"  - 输出类别数: 120")
    print(f"  - 微调方式: 冻结特征提取层，训练新的输出层")
    
    print("\n训练参数:")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 训练轮数: {num_epochs}")
    print(f"  - 学习率: {lr}")
    print(f"  - 权重衰减: {wd}")
    print(f"  - 学习率衰减周期: {lr_period}")
    print(f"  - 学习率衰减率: {lr_decay}")
    print(f"  - 设备: {devices}")
