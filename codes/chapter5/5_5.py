import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
from torch.nn import functional as F  # 导入PyTorch函数式接口


# 5.5.1 加载和保存张量
x = torch.randn(4)  # 创建一个包含4个随机数的张量
torch.save(x, "x-file")  # 将张量保存到文件

x2 = torch.load("x-file")  # 从文件加载张量
print("x2:", x2)  # 打印加载的张量

y = torch.zeros(4)  # 创建一个包含4个零的张量
torch.save([x, y], "x-files")  # 将两个张量保存到文件

x2, y2 = torch.load("x-files")  # 从文件加载两个张量
print("x2:", x2)  # 打印加载的第一个张量
print("y2:", y2)  # 打印加载的第二个张量

mydict = {"x": x, "y": y}  # 创建一个包含张量的字典
torch.save(mydict, "mydict")  # 将字典保存到文件

mydict2 = torch.load("mydict")  # 从文件加载字典
print("mydict2:", mydict2)  # 打印加载的字典

# 5.5.2 加载和保存模型参数
class MLP(nn.Module):  # 定义多层感知机类
    def __init__(self):  # 初始化函数
        super().__init__()  # 调用父类的初始化函数
        self.hidden = nn.Linear(20, 256)  # 定义隐藏层，输入维度20，输出维度256
        self.output = nn.Linear(256, 10)  # 定义输出层，输入维度256，输出维度10
    def forward(self, X):  # 前向传播函数
        return self.output(F.relu(self.hidden(X)))  # 先通过隐藏层，再经过ReLU激活，最后通过输出层

net = MLP()  # 创建MLP模型实例
X = torch.randn(2, 20)  # 创建一个2x20的随机输入张量
Y = net(X)  # 前向传播，计算输出

print("net(X):", net(X))  # 打印网络的输出

torch.save(net.state_dict(), "mlp.params")  # 保存模型的参数字典到文件

clone = MLP()  # 创建一个新的MLP模型实例
clone.load_state_dict(torch.load("mlp.params"))  # 从文件加载参数到新模型
print("clone.eval():", clone.eval())  # 将模型设置为评估模式并打印

Y_clone = clone(X)  # 使用克隆的模型计算输出
print("Y_clone == Y:", Y_clone == Y)  # 比较克隆模型的输出与原模型的输出
