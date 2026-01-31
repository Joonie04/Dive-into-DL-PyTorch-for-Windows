import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块

class CenteredLayer(nn.Module):  # 定义中心化层类
    def __init__(self):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数

    def forward(self, X):  # 前向传播函数
        return X - X.mean()  # 返回输入减去其均值，实现中心化

layer = CenteredLayer()  # 创建中心化层实例
print("layer:", layer(torch.FloatTensor([1, 2, 3, 4, 5])))  # 打印中心化层的输出

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())  # 创建一个包含线性层和中心化层的网络

Y = net(torch.rand(4, 8))  # 生成随机输入并通过网络
print("Y.mean():", Y.mean())  # 打印输出的均值，应该接近0

class MyLinear(nn.Module):  # 定义自定义线性层类
    def __init__(self, in_units, units):  # 初始化函数，接收输入维度和输出维度
        super().__init__()  # 调用父类的初始化函数
        self.weight = nn.Parameter(torch.randn(in_units, units))  # 定义权重参数
        self.bias = nn.Parameter(torch.randn(units,))  # 定义偏置参数
    def forward(self, X):  # 前向传播函数
        linear = torch.matmul(X, self.weight.data) + self.bias.data  # 计算线性变换
        return nn.functional.relu(linear)  # 应用ReLU激活函数

linear = MyLinear(5, 3)  # 创建自定义线性层实例，输入维度5，输出维度3
print("linear.weight:", linear.weight)  # 打印权重
print("linear.torch:", linear(torch.rand(2, 5)))  # 打印随机输入的输出

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))  # 创建一个包含两个自定义线性层的网络
print("net:", net(torch.rand(2, 64)))  # 打印随机输入的输出
