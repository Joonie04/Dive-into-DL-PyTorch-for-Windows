import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
from torch.nn import functional as F  # 导入PyTorch函数式接口

# 定义一个简单的神经网络
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))  # 创建一个包含两个线性层和ReLU激活函数的顺序模型

X = torch.randn(2, 20)  # 创建一个2x20的随机输入张量
print("net(X):", net(X))  # 打印网络的输出

# 5.1.1 自定义块
class MLP(nn.Module):  # 定义多层感知机类，继承自nn.Module
    def __init__(self):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.hidden = nn.Linear(20, 256)  # 定义隐藏层，输入维度20，输出维度256
        self.out = nn.Linear(256, 10)  # 定义输出层，输入维度256，输出维度10

    def forward(self, X):  # 前向传播函数
        return self.out(F.relu(self.hidden(X)))  # 先通过隐藏层，再经过ReLU激活，最后通过输出层

net = MLP()  # 创建MLP模型实例
print("自定义块:", net(X))  # 打印自定义块的输出

# 5.1.2 顺序块
class MySequential(nn.Module):  # 定义自定义顺序块类
    def __init__(self, *args):  # 初始化函数，接收可变数量的模块
        super().__init__()  # 调用父类的初始化函数
        for idx, module in enumerate(args):  # 遍历所有传入的模块
            self.add_module(str(idx), module)  # 将模块添加到模型中，使用索引作为名称

    def forward(self, X):  # 前向传播函数
        for module in self.children():  # 遍历所有子模块
            X = module(X)  # 依次通过每个模块
        return X  # 返回最终输出
    
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))  # 创建自定义顺序块
print("顺序块:", net(X))  # 打印顺序块的输出

# 5.1.3 自定义前向传播
class FixedHiddenMLP(nn.Module):  # 定义固定隐藏层MLP类
    def __init__(self):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.rand_weight = torch.rand((20, 20), requires_grad=False)  # 创建一个固定的随机权重矩阵，不需要梯度
        self.linear = nn.Linear(20, 20)  # 定义线性层

    def forward(self, X):  # 前向传播函数
        X = self.linear(X)  # 通过线性层
        X = F.relu(torch.mm(X, self.rand_weight) + 1)  # 与固定权重矩阵相乘，加1，然后经过ReLU激活
        X = self.linear(X)  # 再次通过线性层
        return X  # 返回输出
    
net = FixedHiddenMLP()  # 创建固定隐藏层MLP模型实例
print("自定义前向传播:", net(X))  # 打印自定义前向传播的输出

# 5.1.4 混合搭配块
class NestMLP(nn.Module):  # 定义嵌套MLP类
    def __init__(self):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),  # 定义一个嵌套的顺序块
                                  nn.Linear(64, 32), nn.ReLU())  # 包含两个线性层和ReLU激活
        self.linear = nn.Linear(32, 16)  # 定义输出层

    def forward(self, X):  # 前向传播函数
        return self.linear(self.net(X))  # 先通过嵌套的顺序块，再通过输出层
    
net = NestMLP()  # 创建嵌套MLP模型实例
print("混合搭配块:", net(X))  # 打印混合搭配块的输出
