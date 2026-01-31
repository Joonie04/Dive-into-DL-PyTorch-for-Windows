import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块

# 5.6.1 计算设备
print("torch.device('cpu'):", torch.device("cpu"))  # 打印CPU设备对象
print("torch.device('cuda'):", torch.device("cuda"))  # 打印CUDA设备对象
print("torch.device('cuda:0'):", torch.device("cuda:0"))  # 打印第一个CUDA设备对象

print("torch.cuda.device_count():", torch.cuda.device_count())  # 打印可用的CUDA设备数量

def try_gpu(i=0):  # 定义尝试使用GPU的函数
    if torch.cuda.device_count() >= i + 1:  # 如果存在第i个GPU
        return torch.device(f"cuda:{i}")  # 返回第i个GPU设备
    return torch.device("cpu")  # 否则返回CPU设备

def try_all_gpus():  # 定义尝试使用所有GPU的函数
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]  # 获取所有可用的GPU设备
    return devices if devices else [torch.device("cpu")]  # 如果有GPU则返回GPU列表，否则返回CPU设备列表

print("try_gpu():", try_gpu())  # 打印尝试使用GPU的结果
print("try_gpu(10):", try_gpu(10))  # 打印尝试使用第10个GPU的结果
print("try_all_gpus():", try_all_gpus())  # 打印尝试使用所有GPU的结果

# 5.6.2 张量与GPU
X = torch.tensor([1, 2, 3])  # 创建一个张量
print("X.device:", X.device)  # 打印张量所在的设备

# 1. 存储在GPU上
X = torch.randn(2, 3, device=try_gpu())  # 在GPU上创建一个2x3的随机张量
print("X.device:", X.device)  # 打印张量所在的设备
Y = torch.randn(2, 3, device=try_gpu(0))  # 在第一个GPU上创建一个2x3的随机张量
print("Y.device:", Y.device)  # 打印张量所在的设备

# 2.复制
Z = X.cuda(0)  # 将X复制到第一个GPU
print("X:", X)  # 打印X
print("Z:", Z)  # 打印Z
print("Y + Z:", Y + Z)  # 打印Y和Z的和
print("Z.cuda(1) is Z:", Z.cuda(0) is Z)  # 检查复制到同一GPU是否返回同一对象


# 5.6.3 模型与GPU
net = nn.Sequential(nn.Linear(3, 1))  # 创建一个包含线性层的网络
print("net:", net)  # 打印网络结构
print("net[0].weight.device:", net[0].weight.device)  # 打印权重所在的设备

net = net.to(try_gpu())  # 将模型移动到GPU
print("net[0].weight.device:", net[0].weight.data.device)  # 打印移动后权重所在的设备
