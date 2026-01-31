import torch  # 导入PyTorch库
import torchvision  # 导入torchvision库，用于计算机视觉任务
from torch import nn  # 导入PyTorch神经网络模块
from torch.utils import data  # 导入PyTorch数据处理工具
from torchvision import transforms  # 导入图像变换模块
import sys  # 导入系统模块
import torch  # 导入PyTorch库（重复导入）
import torch.nn as nn  # 导入PyTorch神经网络模块（重复导入）
import torch.nn.functional as F  # 导入PyTorch神经网络函数模块
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot绘图模块
import time  # 导入时间模块

# 轻量级d2l函数实现，避免torchtext依赖问题
def evaluate_accuracy(data_iter, net, device=None):  # 定义评估模型准确率的函数
    if device is None and isinstance(net, torch.nn.Module):  # 如果没有指定设备且是PyTorch模型
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device if list(net.parameters()) else torch.device('cpu')  # 获取模型所在的设备
    acc_sum, n = 0.0, 0  # 初始化准确率总和和样本数
    with torch.no_grad():  # 在不计算梯度的上下文中执行
        for X, y in data_iter:  # 遍历数据迭代器
            if isinstance(net, torch.nn.Module):  # 如果是PyTorch模型
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()  # 计算正确预测数
                net.train() # 改回训练模式
            else: # 自定义的模型
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()  # 计算正确预测数
            n += y.shape[0]  # 累加样本数
    return acc_sum / n  # 返回准确率

def train(net, train_iter, test_iter, loss, num_epochs, batch_size,  # 定义训练模型的函数
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):  # 遍历每个epoch
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0  # 初始化训练损失、训练准确率和样本数
        for X, y in train_iter:  # 遍历训练数据迭代器
            y_hat = net(X)  # 前向传播，计算预测值
            l = loss(y_hat, y).sum()  # 计算损失并求和
            
            # 梯度清零
            if optimizer is not None:  # 如果使用优化器
                optimizer.zero_grad()  # 清零梯度
            elif params is not None and params[0].grad is not None:  # 如果使用自定义参数且有梯度
                for param in params:  # 遍历所有参数
                    param.grad.data.zero_()  # 清零梯度
            
            l.backward()  # 反向传播计算梯度
            if optimizer is None:  # 如果没有使用优化器
                # 使用SGD更新参数
                with torch.no_grad():  # 在不计算梯度的上下文中执行
                    for param in params:  # 遍历所有参数
                        param.data -= lr * param.grad / batch_size  # 使用梯度更新参数
            else:  # 如果使用了优化器
                optimizer.step()  # 使用优化器更新参数
            
            train_l_sum += l.item()  # 累加训练损失
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()  # 累加正确预测数
            n += y.shape[0]  # 累加样本数
        test_acc = evaluate_accuracy(test_iter, net)  # 评估测试集准确率
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'  # 打印训练结果
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))  # 格式化输出


def init_weights(m):  # 定义初始化权重的函数
    if type(m) == nn.Linear:  # 如果是线性层
        nn.init.normal_(m.weight, std=0.01)  # 使用正态分布初始化权重


if __name__ == '__main__':  # 当脚本作为主程序运行时执行
    batch_size = 256  # 设置批次大小
    trans = transforms.ToTensor()  # 定义图像转换为张量的变换
    data_path = 'dataset/FashionMNIST/raw'  # 设置数据集存储路径
    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=trans, download=True)  # 加载训练集
    test_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=trans, download=True)  # 加载测试集
   
    train_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 创建训练数据迭代器
    test_iter = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 创建测试数据迭代器

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))  # 定义softmax回归模型（展平层+线性层）
    net.apply(init_weights)  # 应用权重初始化

    loss = nn.CrossEntropyLoss(reduction='none')  # 定义交叉熵损失函数

    trainer = torch.optim.SGD(net.parameters(), lr=0.1)  # 定义随机梯度下降优化器

    num_epochs = 10  # 设置训练轮数

    train(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer=trainer)  # 训练模型