import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块

# 6.2.1 二维卷积层
def corr2d(X, K):  # 定义二维卷积函数
    h, w = K.shape  # 获取卷积核的高度和宽度
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))  # 创建输出张量，计算输出尺寸
    for i in range(Y.shape[0]):  # 遍历输出的每一行
        for j in range(Y.shape[1]):  # 遍历输出的每一列
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()  # 计算卷积操作：对应位置相乘后求和
    return Y  # 返回卷积结果

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])  # 创建输入张量
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])  # 创建卷积核
print("corr2d(X, K):", corr2d(X, K))  # 打印卷积结果

# 6.2.2 卷积层
class Conv2D(nn.Module):  # 定义二维卷积层类
    def __init__(self, kernel_size):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.weight = nn.Parameter(torch.rand(kernel_size))  # 定义卷积核权重参数
        self.bias = nn.Parameter(torch.zeros(1))  # 定义偏置参数

    def forward(self, x):  # 前向传播函数
        return corr2d(x, self.weight) + self.bias  # 执行卷积操作并加上偏置


# 6.2.3 图像中目标的边缘检测
X = torch.ones((6, 8))  # 创建一个6x8的全1张量
X[:, 2:6] = 0  # 将第2到5列设为0，创建一个垂直边缘
print("X:", X)  # 打印输入图像

K = torch.tensor([[1.0, -1.0]])  # 创建一个水平边缘检测卷积核
print("K:", K)  # 打印卷积核

Y = corr2d(X, K)  # 执行卷积操作
print("Y:", Y)  # 打印卷积结果，检测到的边缘

print("corr2d(X.t(), K):", corr2d(X.t(), K))  # 对转置后的输入执行卷积，检测垂直边缘

# 6.2.4 学习卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)  # 创建一个二维卷积层，1个输入通道，1个输出通道，卷积核大小为1x2

X = X.reshape((1, 1, 6, 8))  # 将输入重塑为4D张量（批量大小，通道数，高度，宽度）
Y = Y.reshape((1, 1, 6, 7))  # 将输出重塑为4D张量
lr = 3e-2  # 设置学习率

print("X:", X)  # 打印输入张量
print("Y:", Y)  # 打印目标输出张量
print("lr:", lr)  # 打印学习率

for i in range(10):  # 训练10次
    Y_hat = conv2d(X)  # 前向传播，计算预测输出
    l = (Y_hat - Y) ** 2  # 计算均方误差损失
    conv2d.zero_grad()  # 清零梯度
    l.sum().backward()  # 反向传播，计算梯度
    conv2d.weight.data[:] -= lr * conv2d.weight.grad  # 更新卷积核权重
    if (i + 1) % 2 == 0:  # 每2次迭代打印一次
        print(f"batch {i + 1}, loss {l.sum():.3f}")  # 打印批次号和损失

print("conv2d.weight.data:", conv2d.weight.data)  # 打印学习到的卷积核权重
