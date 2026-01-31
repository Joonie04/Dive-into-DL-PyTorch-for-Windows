# 13.11 全卷积网络

import os
import sys
import time
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def set_figsize(figsize=(3.5, 2.5)):
    """设置图像大小
    
    参数:
        figsize: 图像大小，格式为 (width, height)
    """
    plt.rcParams['figure.figsize'] = figsize

def show_images(images, num_rows, num_cols, scale=1.5):
    """显示多张图像
    
    参数:
        images: 图像列表或张量
        num_rows: 行数
        num_cols: 列数
        scale: 缩放比例
    
    返回:
        axes 对象列表
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() if num_rows * num_cols > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            if torch.is_tensor(img):
                img = img.permute(1, 2, 0).numpy()
            ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    return axes

def try_gpu(i=0):
    """尝试获取 GPU 设备
    
    参数:
        i: GPU 索引
    
    返回:
        如果 GPU 可用返回 GPU 设备，否则返回 CPU 设备
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')

def try_all_gpus():
    """尝试获取所有可用的 GPU 设备
    
    返回:
        GPU 设备列表，如果没有 GPU 则返回包含 CPU 的列表
    """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [try_gpu()]

class Accumulator:
    """累加器类，用于累加多个指标"""
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    """计时器类，用于测量代码执行时间"""
    def __init__(self):
        self.times = []
        self.start()
    
    def start(self):
        self.tik = time.time()
    
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times) / len(self.times)
    
    def sum(self):
        return sum(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

class Animator:
    """动画可视化类，用于绘制训练过程中的指标变化"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: self.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    
    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
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
        return self._Y if hasattr(self, '_Y') else None
    
    @Y.setter
    def Y(self, value):
        self._Y = value

def accuracy(y_hat, y):
    """计算预测准确率
    
    参数:
        y_hat: 预测值
        y: 真实标签
    
    返回:
        准确率
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(torch.float32).sum())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的准确率
    
    参数:
        net: 神经网络模型
        data_iter: 数据迭代器
        device: 设备
    
    返回:
        准确率和样本数量
    """
    if isinstance(net, nn.Module):
        net.eval()
    if not device:
        device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch13(net, train_iter, test_iter, loss_fn, trainer, num_epochs, devices):
    """训练语义分割模型
    
    参数:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        loss_fn: 损失函数
        trainer: 优化器
        num_epochs: 训练轮数
        devices: 设备列表
    """
    timer, num_batches = Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = None, None
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            outputs = net(features)
            l = loss_fn(outputs, labels)
            l.sum().backward()
            trainer.step()
            with torch.no_grad():
                acc = accuracy(outputs, labels)
            metric.add(l.sum(), l.numel(), acc, labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[1], metric[2] / metric[3], None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[1]:.3f}, train acc {metric[2] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')

def download_voc2012():
    """下载 Pascal VOC2012 数据集
    
    返回:
        数据集目录路径
    """
    dataset_dir = "dataset/voc2012"
    voc_dir = os.path.join(dataset_dir, "VOCdevkit", "VOC2012")
    
    if os.path.exists(voc_dir):
        print(f"数据集已存在于 {voc_dir}")
        return voc_dir
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from downloader.voc2012 import download_voc2012 as download
    return download()

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注
    
    参数:
        voc_dir: VOC数据集目录路径
        is_train: 是否为训练集，True 表示训练集，False 表示验证集
    
    返回:
        features: 图像列表，每个图像为张量
        labels: 标签列表，每个标签为张量
    """
    txt_fname = os.path.join(voc_dir, 'ImageSets/Segmentation/' + ('train.txt' if is_train else 'val.txt'))
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for fname in images:
        features.append(torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'), mode))
    return features, labels

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128], [128, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射
    
    返回:
        colormap2label: 形状为 (256^3,) 的张量，将RGB值映射到类别索引
    """
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引
    
    参数:
        colormap: 标签图像，形状为 (3, height, width)
        colormap2label: RGB到类别索引的映射
    
    返回:
        类别索引张量，形状为 (height, width)
    """
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像
    
    参数:
        feature: 特征图像，形状为 (3, h, w)
        label: 标签图像，形状为 (3, h, w)
        height: 裁剪高度
        width: 裁剪宽度
    
    返回:
        (cropped_feature, cropped_label) 裁剪后的特征和标签
    """
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集
    
    参数:
        is_train: 是否为训练集
        crop_size: 裁剪尺寸 (height, width)
        voc_dir: VOC数据集目录路径
    """
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature) for feature in features]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        """归一化图像
        
        参数:
            img: 图像张量，形状为 (3, h, w)
        
        返回:
            归一化后的图像
        """
        return self.transform(img.float() / 255)

    def filter(self, labels):
        """过滤掉尺寸小于crop_size的标签
        
        参数:
            labels: 标签列表
        
        返回:
            过滤后的标签列表
        """
        return [label for label in labels if (
            label.shape[1] >= self.crop_size[0] and
            label.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        """获取单个样本
        
        参数:
            idx: 样本索引
        
        返回:
            (feature, label) 元组
            feature: 归一化后的特征图像
            label: 类别索引标签
        """
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],*self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        """返回数据集大小
        
        返回:
            数据集样本数量
        """
        return len(self.features)

def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集
    
    参数:
        batch_size: 批次大小
        crop_size: 裁剪尺寸 (height, width)
    
    返回:
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
    """
    voc_dir = download_voc2012()
    num_workers = 0
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        num_workers=num_workers)
    return train_iter, test_iter

## 13.11.1 构建模型
pertrained_net = torchvision.models.resnet18(pretrained=True)
print("list(pertrained_net.children())[-3:]", list(pertrained_net.children())[-3:])

net = nn.Sequential(*list(pertrained_net.children())[:-2])
X = torch.rand(size=(1, 3, 320, 480))
print("net(X).shape", net(X).shape)

num_classes = 21
net.add_module("final_conv", nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module("transpose_conv", nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))

## 13.11.2 初始化转置卷积层
def bilinear_kernel(in_channels, out_channels, kernel_size):
    """生成双线性插值核
    
    双线性插值是一种常用的图像上采样方法，它可以保持图像的平滑性。
    这个函数生成用于转置卷积的双线性插值核。
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
    
    返回:
        weight: 卷积核权重，形状为 (in_channels, out_channels, kernel_size, kernel_size)
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

img_path = "dataset/img/catdog.jpg"
if os.path.exists(img_path):
    img = torchvision.transforms.ToTensor()(Image.open(img_path))
else:
    print(f"图像文件不存在: {img_path}")
    print("使用随机生成的图像进行演示")
    img = torch.rand(3, 256, 256)

X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
set_figsize()
print("input image shape:", img.permute(1, 2, 0).shape)
plt.imshow(img.permute(1, 2, 0))
plt.show()
print("output image shape:", out_img.shape)
plt.imshow(out_img)
plt.show()

W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

## 13.11.3 读取数据集
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = load_data_voc(batch_size, crop_size)

## 13.11.4 训练
def loss(inputs, targets):
    """计算交叉熵损失
    
    参数:
        inputs: 模型输出，形状为 (batch_size, num_classes, height, width)
        targets: 真实标签，形状为 (batch_size, height, width)
    
    返回:
        损失值，形状为 (batch_size,)
    """
    return F.cross_entropy(inputs, targets, reduction="none").mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

## 13.11.5 预测
def predict(img):
    """预测图像的语义分割结果
    
    参数:
        img: 输入图像，形状为 (3, height, width)
    
    返回:
        pred: 预测的类别索引，形状为 (height, width)
    """
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    """将类别索引转换为彩色图像
    
    参数:
        pred: 类别索引张量，形状为 (height, width)
    
    返回:
        彩色图像，形状为 (height, width, 3)
    """
    colormap = torch.tensor(VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]

voc_dir = download_voc2012()
test_images, test_labels = read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1, 2, 0), pred.cpu(),
             torchvision.transforms.functional.crop(test_labels[i], *crop_rect).permute(1, 2, 0)]
show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("全卷积网络（FCN）用于语义分割")
    print("=" * 60)
    
    print("\n模型结构:")
    print("  - 基础网络: ResNet-18（预训练）")
    print("  - 最终卷积层: 1x1 卷积，将 512 通道映射到 21 个类别")
    print("  - 转置卷积层: 64x64 卷积核，stride=32，用于上采样")
    
    print("\n数据集类别:")
    for i, cls in enumerate(VOC_CLASSES):
        print(f"  {i:2d}: {cls}")
