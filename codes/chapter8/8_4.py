# 8.4 循环神经网络

import torch  # 导入PyTorch库

# 初始化输入和隐藏状态
X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))  # X: 输入向量，形状为(3, 1)；W_xh: 输入到隐藏状态的权重矩阵，形状为(1, 4)
H, W_hh = torch.normal(0, 1, (1, 4)), torch.normal(0, 1, (4, 4))  # H: 隐藏状态，形状为(1, 4)；W_hh: 隐藏状态到隐藏状态的权重矩阵，形状为(4, 4)

# 方法1：分别计算输入到隐藏状态和隐藏状态到隐藏状态的矩阵乘法，然后相加
print("torch.matmul(X, W_xh) + torch.matmul(H, W_hh)", torch.matmul(X, W_xh) + torch.matmul(H, W_hh))  # 打印：X*W_xh + H*W_hh，这是RNN的标准计算方式

# 方法2：将输入和隐藏状态拼接，将权重矩阵拼接，然后进行一次矩阵乘法
torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))  # 拼接X和H（在维度1上），拼接W_xh和W_hh（在维度0上），然后进行矩阵乘法
print("torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))", torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))  # 打印：[X, H] * [W_xh; W_hh]，这是等价的计算方式
