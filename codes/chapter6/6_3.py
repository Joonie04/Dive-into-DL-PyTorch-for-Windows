import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块

def comp_conv2d(conv2d, X):  # 定义计算卷积输出形状的函数
    X = X.reshape((1, 1) + X.shape)  # 将输入重塑为4D张量（批量大小，通道数，高度，宽度）
    Y = conv2d(X)  # 执行卷积操作
    return Y.reshape(Y.shape[2:])  # 返回输出的空间维度（高度和宽度）

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # 创建卷积层：1输入通道，1输出通道，3x3卷积核，填充1
X = torch.rand(size=(8, 8))  # 创建一个8x8的随机输入张量
print("comp_conv2d(conv2d, X).shape 1:", comp_conv2d(conv2d, X).shape)  # 打印输出形状，应该与输入相同（8x8）

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))  # 创建卷积层：5x3卷积核，填充分别为2和1
print("comp_conv2d(conv2d, X).shape 2:", comp_conv2d(conv2d, X).shape)  # 打印输出形状，应该与输入相同（8x8）

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=2, stride=2)  # 创建卷积层：3x3卷积核，填充2，步幅2
print("comp_conv2d(conv2d, X).shape 3:", comp_conv2d(conv2d, X).shape)  # 打印输出形状，应该是输入的一半（4x4）

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))  # 创建卷积层：3x5卷积核，填充分别为0和1，步幅分别为3和4
print("comp_conv2d(conv2d, X).shape 4:", comp_conv2d(conv2d, X).shape)  # 打印输出形状，高度和宽度都减小
