# 13.12 风格迁移

## 13.12.1 方法

## 13.12.2 读取内容和风格图像

import os
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image

def set_figsize(figsize=(3.5, 2.5)):
    """设置图像大小
    
    参数:
        figsize: 图像大小，格式为 (width, height)
    """
    plt.rcParams['figure.figsize'] = figsize

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

set_figsize()

content_img_path = "dataset/img/rainier.jpg"
if os.path.exists(content_img_path):
    content_img = Image.open(content_img_path)
    print("内容图像:")
    plt.imshow(content_img)
    plt.show()
else:
    print(f"内容图像不存在: {content_img_path}")
    print("使用随机生成的图像进行演示")
    content_img = torchvision.transforms.ToPILImage()(torch.rand(3, 300, 450))

style_img_path = "dataset/img/autumn-oak.jpg"
if os.path.exists(style_img_path):
    style_img = Image.open(style_img_path)
    print("风格图像:")
    plt.imshow(style_img)
    plt.show()
else:
    print(f"风格图像不存在: {style_img_path}")
    print("使用随机生成的图像进行演示")
    style_img = torchvision.transforms.ToPILImage()(torch.rand(3, 300, 450))

## 13.12.3 预处理和后处理

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    """预处理图像
    
    将图像调整为指定大小，转换为张量，并进行归一化。
    
    参数:
        img: PIL 图像对象
        image_shape: 目标图像尺寸 (height, width)
    
    返回:
        归一化后的张量，形状为 (1, 3, height, width)
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)
    ])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    """后处理图像
    
    将张量转换回 PIL 图像，并反归一化。
    
    参数:
        img: 张量，形状为 (1, 3, height, width)
    
    返回:
        PIL 图像对象
    """
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

## 13.12.4 提取图像特征

pretrained_net = torchvision.models.vgg19(pretrained=True)
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
net = nn.Sequential(*[pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)])

def extract_features(X, content_layers, style_layers):
    """提取图像的内容特征和风格特征
    
    通过预训练的 VGG-19 网络提取图像的多层特征。
    内容层用于保留图像的内容信息，风格层用于捕获图像的风格信息。
    
    参数:
        X: 输入图像张量
        content_layers: 内容层的索引列表
        style_layers: 风格层的索引列表
    
    返回:
        contents: 内容特征列表
        styles: 风格特征列表
    """
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def get_contents(image_shape, device):
    """获取内容图像的特征
    
    参数:
        image_shape: 图像尺寸 (height, width)
        device: 计算设备
    
    返回:
        content_X: 预处理后的内容图像
        contents_Y: 内容特征列表
    """
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    """获取风格图像的特征
    
    参数:
        image_shape: 图像尺寸 (height, width)
        device: 计算设备
    
    返回:
        style_X: 预处理后的风格图像
        styles_Y: 风格特征列表
    """
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

## 13.12.5 定义损失函数

### 1. 内容损失
def content_loss(Y_hat, Y):
    """计算内容损失
    
    内容损失用于衡量生成图像与内容图像在内容上的相似度。
    使用均方误差作为损失函数。
    
    参数:
        Y_hat: 生成图像的特征
        Y: 内容图像的特征
    
    返回:
        内容损失值
    """
    return torch.square(Y_hat - Y.detach()).mean()

### 2. 风格损失
def gram(X):
    """计算格拉姆矩阵
    
    格拉姆矩阵用于表示图像的风格信息，它捕获了特征图之间的相关性。
    格拉姆矩阵的计算方式为：G = X * X^T / (C * N)
    其中 C 是通道数，N 是特征元素总数。
    
    参数:
        X: 特征张量，形状为 (batch, channels, height, width)
    
    返回:
        格拉姆矩阵，形状为 (channels, channels)
    """
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    """计算风格损失
    
    风格损失用于衡量生成图像与风格图像在风格上的相似度。
    通过比较生成图像和风格图像的格拉姆矩阵来计算。
    
    参数:
        Y_hat: 生成图像的特征
        gram_Y: 风格图像的格拉姆矩阵
    
    返回:
        风格损失值
    """
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

### 3. 全变分损失
def tv_loss(Y_hat):
    """计算全变分损失
    
    全变分损失用于鼓励生成图像的平滑性，减少噪声。
    它计算图像在水平和垂直方向上的梯度总和。
    
    参数:
        Y_hat: 生成图像
    
    返回:
        全变分损失值
    """
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

### 4. 损失函数
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    """计算总损失
    
    总损失是内容损失、风格损失和全变分损失的加权和。
    
    参数:
        X: 生成图像
        contents_Y_hat: 生成图像的内容特征
        styles_Y_hat: 生成图像的风格特征
        contents_Y: 内容图像的内容特征
        styles_Y_gram: 风格图像的格拉姆矩阵
    
    返回:
        contents_l: 内容损失列表
        styles_l: 风格损失列表
        tv_l: 全变分损失
        l: 总损失
    """
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

## 13.12.6 初始化合成图像

class SynthesizedImage(nn.Module):
    """合成图像类
    
    这个类将图像像素作为可训练参数，通过梯度下降来优化图像。
    
    参数:
        img_shape: 图像形状 (1, 3, height, width)
    """
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
    """初始化合成图像和优化器
    
    参数:
        X: 初始图像张量
        device: 计算设备
        lr: 学习率
        styles_Y: 风格特征列表
    
    返回:
        gen_img: 合成图像模型
        styles_Y_gram: 风格图像的格拉姆矩阵列表
        trainer: 优化器
    """
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

## 13.12.7 训练模型

def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    """训练风格迁移模型
    
    通过优化合成图像的像素值，使得生成图像在内容上接近内容图像，
    在风格上接近风格图像。
    
    参数:
        X: 初始图像张量
        contents_Y: 内容图像的特征
        styles_Y: 风格图像的特征
        device: 计算设备
        lr: 初始学习率
        num_epochs: 训练轮数
        lr_decay_epoch: 学习率衰减周期
    
    返回:
        X: 训练后的合成图像
    """
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)), float(sum(styles_l)), float(tv_l)])
    
    return X

device, image_shape = try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)

if __name__ == "__main__":
    print("=" * 60)
    print("风格迁移（Neural Style Transfer）")
    print("=" * 60)
    
    print("\n方法说明:")
    print("  风格迁移通过优化合成图像的像素值，")
    print("  使得生成图像在内容上接近内容图像，")
    print("  在风格上接近风格图像。")
    
    print("\n损失函数:")
    print(f"  - 内容损失权重: {content_weight}")
    print(f"  - 风格损失权重: {style_weight}")
    print(f"  - 全变分损失权重: {tv_weight}")
    
    print("\n网络结构:")
    print("  - 基础网络: VGG-19（预训练）")
    print("  - 内容层: [25]")
    print("  - 风格层: [0, 5, 10, 19, 28]")
