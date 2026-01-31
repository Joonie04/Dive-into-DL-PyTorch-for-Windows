import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
from torch.utils import data  # 导入PyTorch数据处理工具
from torch import nn  # 导入PyTorch神经网络模块


def synthetic_data(w, b, num_examples):  # 定义生成合成数据的函数
    """生成y=Xw+b+噪声"""  # 函数文档字符串
    X = torch.normal(0, 1, (num_examples, len(w)))  # 生成均值为0、标准差为1的正态分布特征矩阵
    y = torch.matmul(X, w) + b  # 计算线性回归的输出 y = Xw + b
    y += torch.normal(0, 0.01, y.shape)  # 添加均值为0、标准差为0.01的高斯噪声
    return X, y.reshape((-1, 1))  # 返回特征矩阵X和重塑后的标签y


def load_array(data_arrays, batch_size, is_train=True):  # 定义加载数据的函数
    """
    构造一个PyTorch数据迭代器
    """  # 函数文档字符串
    dataset = data.TensorDataset(*data_arrays)  # 创建PyTorch张量数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 返回数据加载器，可选择是否打乱数据


if __name__ == '__main__':  # 当脚本作为主程序运行时执行

    
    true_w = torch.tensor([2, -3.4])  # 设置真实的权重参数
    true_b = 4.2  # 设置真实的偏置参数
    features, labels = synthetic_data(true_w, true_b, 1000)  # 生成1000个样本的合成数据

    batch_size = 10  # 设置批次大小
    data_iter = load_array((features, labels), batch_size)  # 创建数据迭代器

    net = nn.Sequential(nn.Linear(2, 1))  # 定义线性回归模型（输入维度2，输出维度1）
    loss = nn.MSELoss()  # 定义均方误差损失函数

    trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 定义随机梯度下降优化器，学习率为0.03

    num_epochs = 3  # 设置训练轮数
    for epoch in range(num_epochs):  # 遍历每一轮训练
        for X, y in data_iter:  # 遍历每个小批量数据
            l = loss(net(X), y)  # 计算预测值和真实值之间的损失
            trainer.zero_grad()  # 清零梯度
            l.backward()  # 反向传播计算梯度
            trainer.step()  # 使用梯度更新参数
        l = loss(net(features), labels)  # 计算整个数据集上的损失
        print(f'epoch {epoch + 1}, loss {l:f}')  # 打印当前轮次和损失
        w = net[0].weight.data  # 获取学习到的权重参数
        print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')  # 打印权重参数的估计误差
        b = net[0].bias.data  # 获取学习到的偏置参数
        print(f'b的估计误差: {true_b - b}')  # 打印偏置参数的估计误差
