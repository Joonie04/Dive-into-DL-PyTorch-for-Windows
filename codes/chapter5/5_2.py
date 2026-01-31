# 参数管理

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))  # 创建一个包含两个线性层和ReLU激活函数的顺序模型
X = torch.randn(2, 4)  # 创建一个2x4的随机输入张量
print("net(X):", net(X))  # 打印网络的输出

# 5.2.1 访问参数
print("net[2].state_dict():", net[2].state_dict())  # 打印第三个模块（第二个线性层）的状态字典

# 1. 目标参数
print("type(net[2].bias):", type(net[2].bias))  # 打印偏置参数的类型
print("net[2].bias:", net[2].bias)  # 打印偏置参数的值
print("net[2].bias.grad:", net[2].bias.grad)  # 打印偏置参数的梯度

print("net[2].weight.grad == None:", net[2].weight.grad == None)  # 检查权重梯度是否为None

# 2. 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # 打印第一个模块的所有参数名称和形状
print(*[(name, param.shape) for name, param in net.named_parameters()])  # 打印整个网络的所有参数名称和形状

print("net.state_dict()['2.bias'].data:", net.state_dict()['2.bias'].data)  # 打印第三个模块偏置的数据

# 3. 从嵌套块收集参数
def block1():  # 定义第一个块函数
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())  # 返回一个包含两个线性层和ReLU激活的顺序块

def block2():  # 定义第二个块函数
    net = nn.Sequential()  # 创建一个空的顺序块
    for i in range(4):  # 迭代4次
        net.add_module(f"block {i}", block1())  # 添加4个block1到顺序块中
    return net  # 返回顺序块

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))  # 创建一个包含block2和一个线性层的网络
print("rgnet:", rgnet)  # 打印网络结构

print("rgnet[0][1].bias.data:", rgnet[0][1][0].bias.data)  # 打印嵌套块中特定层的偏置数据

# 5.2.2 参数初始化

# 1. 内置初始化
def init_normal(m):  # 定义正态分布初始化函数
    if type(m) == nn.Linear:  # 如果是线性层
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 使用均值为0、标准差为0.01的正态分布初始化权重
        nn.init.zeros_(m.bias)  # 将偏置初始化为0

net.apply(init_normal)  # 应用正态分布初始化到整个网络
print("net[0].weight:", net[0].weight)  # 打印第一个线性层的权重
print("net[0].bias:", net[0].bias)  # 打印第一个线性层的偏置

def init_constant(m):  # 定义常数初始化函数
    if type(m) == nn.Linear:  # 如果是线性层
        nn.init.constant_(m.weight, 1)  # 将权重初始化为1
        nn.init.zeros_(m.bias)  # 将偏置初始化为0

net.apply(init_constant)  # 应用常数初始化到整个网络
print("net[0].weight:", net[0].weight)  # 打印第一个线性层的权重
print("net[0].bias:", net[0].bias)  # 打印第一个线性层的偏置

def init_xavier(m):  # 定义Xavier初始化函数
    if type(m) == nn.Linear:  # 如果是线性层
        nn.init.xavier_uniform_(m.weight)  # 使用Xavier均匀分布初始化权重

def init_42(m):  # 定义常数42初始化函数
    if type(m) == nn.Linear:  # 如果是线性层
        nn.init.constant_(m.weight, 42)  # 将权重初始化为42

net[0].apply(init_xavier)  # 对第一个线性层应用Xavier初始化
net[2].apply(init_42)  # 对第三个线性层应用常数42初始化
print("net[0].weight:", net[0].weight.data[0])  # 打印第一个线性层权重的第一行
print("net[2].weight:", net[2].weight.data)  # 打印第三个线性层的权重


# 2. 自定义初始化
def my_init(m):  # 定义自定义初始化函数
    if type(m) == nn.Linear:  # 如果是线性层
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()])  # 打印参数信息
        nn.init.uniform_(m.weight, -10, 10)  # 使用-10到10之间的均匀分布初始化权重
        m.weight.data *= m.weight.data.abs() >= 5  # 保留绝对值大于等于5的权重，其他设为0

net.apply(my_init)  # 应用自定义初始化到整个网络
print("net[0].weight:", net[0].weight[:2])  # 打印第一个线性层权重的前两行

net[0].weight.data[:] += 1  # 将第一个线性层的所有权重加1
net[0].weight.data[0, 0] = 42  # 将第一个线性层权重的[0,0]位置设为42
print("net[0].weight:", net[0].weight.data[0])  # 打印第一个线性层权重的第一行

# 5.2.3 参数绑定
shared = nn.Linear(8, 8)  # 创建一个共享的线性层
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),  # 创建一个网络
                    shared, nn.ReLU(),  # 使用共享的线性层
                    shared, nn.ReLU(),  # 再次使用同一个共享的线性层
                    nn.Linear(8, 1))  # 最后一个线性层
net(X)  # 执行一次前向传播
print(net[2].weight.data[0] == net[4].weight.data[0])  # 检查两个共享层的权重是否相同
net[2].weight.data[0, 0] = 100  # 修改第一个共享层的权重
print(net[2].weight.data[0] == net[4].weight.data[0])  # 再次检查，验证参数绑定
