import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块

# 6.4.1 多输入通道
def corr2d_multi_in(X, K):  # 定义多输入通道的二维卷积函数
    return sum(nn.functional.conv2d(x.unsqueeze(0), k.unsqueeze(0)) for x, k in zip(X, K))  # 对每个输入通道分别卷积后求和

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],  # 创建第一个输入通道
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])  # 创建第二个输入通道
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])  # 创建两个卷积核，分别对应两个输入通道

print("corr2d_multi_in(X, K):", corr2d_multi_in(X, K))  # 打印多输入通道卷积结果

# 6.4.2 多输出通道
def corr2d_multi_in_out(X, K):  # 定义多输入多输出通道的二维卷积函数
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)  # 对每个输出通道分别计算卷积，然后堆叠

K = torch.stack((K, K + 1, K + 2), 0)  # 创建3个输出通道的卷积核
print("K.shape:", K.shape)  # 打印卷积核的形状（输出通道数，输入通道数，高度，宽度）
print("corr2d_multi_in_out(X, K):", corr2d_multi_in_out(X, K))  # 打印多输出通道卷积结果

# 6.4.3 1x1卷积层
def corr2d_multi_in_out_1x1(X, K):  # 定义1x1卷积的函数（高效实现）
    c_i, h, w = X.shape  # 获取输入通道数、高度和宽度
    c_o = K.shape[0]  # 获取输出通道数
    X = X.reshape((c_i, h * w))  # 将输入重塑为（输入通道数，像素数）
    K = K.reshape((c_o, c_i))  # 将卷积核重塑为（输出通道数，输入通道数）
    Y = torch.matmul(K, X)  # 使用矩阵乘法计算卷积
    return Y.reshape((c_o, h, w))  # 将输出重塑为（输出通道数，高度，宽度）

X = torch.normal(0, 1, (3, 3, 3))  # 创建一个3通道3x3的随机输入张量
K = torch.normal(0, 1, (2, 3, 1, 1))  # 创建一个2输出通道、3输入通道、1x1的随机卷积核

Y1 = corr2d_multi_in_out_1x1(X, K)  # 使用1x1卷积的高效实现计算
Y2 = corr2d_multi_in_out(X, K)  # 使用常规卷积实现计算

print("Y1:", Y1)  # 打印1x1卷积的结果
print("Y2:", Y2)  # 打印常规卷积的结果

assert float(torch.abs(Y1 - Y2).sum()) < 1e-6  # 断言两种实现的结果差异小于1e-6
