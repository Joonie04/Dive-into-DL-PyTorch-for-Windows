import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块

# 6.5.1 最大汇集和平均汇集
def pool2d(X, pool_size, mode='max'):  # 定义二维池化函数，支持最大池化和平均池化
    p_h, p_w = pool_size  # 获取池化窗口的高度和宽度
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))  # 创建输出张量，计算输出尺寸
    for i in range(Y.shape[0]):  # 遍历输出的每一行
        for j in range(Y.shape[1]):  # 遍历输出的每一列
            if mode == 'max':  # 如果模式是最大池化
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()  # 取窗口内的最大值
            elif mode == 'avg':  # 如果模式是平均池化
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()  # 取窗口内的平均值
    return Y  # 返回池化结果

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])  # 创建一个3x3的输入张量
print("X:", X)  # 打印输入张量
print("pool2d(X, (2, 2)):", pool2d(X, (2, 2)))  # 打印2x2最大池化结果
print("pool2d(X, (2, 2), 'avg'):", pool2d(X, (2, 2), 'avg'))  # 打印2x2平均池化结果

# 6.5.2 填充和步幅
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))  # 创建一个4x4的输入张量，形状为（批量大小，通道数，高度，宽度）
print("X:", X)  # 打印输入张量

pool2d = nn.MaxPool2d(3)  # 创建3x3的最大池化层，默认步幅与池化窗口大小相同
print("pool2d(X):", pool2d(X))  # 打印池化结果

pool2d = nn.MaxPool2d(3, padding=1, stride=2)  # 创建3x3的最大池化层，填充1，步幅2
print("pool2d(X):", pool2d(X))  # 打印池化结果

pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))  # 创建2x3的最大池化层，步幅分别为2和3，填充分别为0和1
print("pool2d(X):", pool2d(X))  # 打印池化结果

# 6.5.3 多个通道
X= torch.cat((X, X + 1), 1)  # 在通道维度上拼接两个张量，创建多通道输入
print("X:", X)  # 打印多通道输入张量

pool2d = nn.MaxPool2d(3, padding=1, stride=2)  # 创建3x3的最大池化层，填充1，步幅2
print("pool2d(X):", pool2d(X))  # 打印多通道池化结果
