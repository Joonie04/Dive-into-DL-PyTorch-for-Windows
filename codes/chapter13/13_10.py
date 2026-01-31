# 13.10 转置卷积

import torch
from torch import nn

## 13.10.1 基本操作
def trans_conv(X, K):
    """转置卷积（反卷积）的基本实现
    
    转置卷积是卷积的一种反向操作，它可以将较小的特征图上采样到较大的尺寸。
    与普通卷积不同，转置卷积的每个输入元素会与卷积核相乘，然后加到输出特征图的多个位置上。
    
    参数:
        X: 输入张量，形状为 (batch_size, channels, height, width)
        K: 卷积核，形状为 (kernel_height, kernel_width)
    
    返回:
        Y: 输出张量，形状为 (batch_size, channels, height + kernel_height - 1, width + kernel_width - 1)
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0], X.shape[1], X.shape[2] + h - 1, X.shape[3] + w - 1))
    for i in range(X.shape[2]):
        for j in range(X.shape[3]):
            Y[:, :, i:i+h, j:j+w] += X[:, :, i, j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print("trans_conv(X, K):", trans_conv(X, K))

## 13.10.2 填充, 步幅和多通道

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print("tconv(X) (padding=1):", tconv(X))

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print("tconv(X) (stride=2):", tconv(X))

X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print("tconv(conv(X)).shape == X.shape:", tconv(conv(X)).shape == X.shape)

## 13.10.3 与矩阵变换的联系

def corr2d(X, K):
    """计算二维互相关（卷积操作）
    
    互相关是卷积神经网络中的基本操作，它通过滑动窗口的方式计算输入和卷积核之间的相关性。
    
    参数:
        X: 输入张量，形状为 (height, width)
        K: 卷积核，形状为 (kernel_height, kernel_width)
    
    返回:
        Y: 输出张量，形状为 (height - kernel_height + 1, width - kernel_width + 1)
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = corr2d(X, K)
print("Y (corr2d output):", Y)

def kernel2matrix(K):
    """将卷积核转换为矩阵形式
    
    这个函数将 2x2 的卷积核转换为 4x9 的稀疏矩阵，
    使得卷积操作可以表示为矩阵乘法。
    
    参数:
        K: 卷积核，形状为 (2, 2)
    
    返回:
        W: 矩阵形式的卷积核，形状为 (4, 9)
    """
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
print("W (kernel matrix):", W)
print("Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2):", Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))

Z = trans_conv(Y, K)
print("Z (trans_conv output):", Z)
print("Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3):", Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))

if __name__ == "__main__":
    print("=" * 60)
    print("转置卷积（反卷积）演示")
    print("=" * 60)
    
    print("\n1. 基本转置卷积操作")
    print("输入 X:")
    print(X)
    print("卷积核 K:")
    print(K)
    print("转置卷积输出:")
    print(trans_conv(X, K))
    
    print("\n2. 填充和步幅的影响")
    print("padding=1 时，输出尺寸增大")
    print("stride=2 时，输出尺寸翻倍")
    
    print("\n3. 卷积与转置卷积的可逆性")
    print("当卷积和转置卷积使用相同的参数时，")
    print("转置卷积可以近似恢复原始输入的形状")
