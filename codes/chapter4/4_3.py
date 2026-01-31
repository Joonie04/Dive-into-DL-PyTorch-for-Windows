import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
import torchvision  # 导入torchvision库，用于计算机视觉任务
import torchvision.transforms as transforms  # 导入图像变换模块
import sys  # 导入系统模块
import os  # 导入操作系统模块

# 设置数据集路径
dataset_path = 'dataset/FashionMNIST/raw'  # 设置FashionMNIST数据集存储路径

# 加载FashionMNIST数据集
def load_data_fashion_mnist(batch_size, resize=None):  # 定义加载FashionMNIST数据集的函数
    trans = []  # 初始化变换列表
    if resize:  # 如果需要调整大小
        trans.append(transforms.Resize(size=resize))  # 添加调整大小的变换
    trans.append(transforms.ToTensor())  # 添加转换为张量的变换
    
    transform = transforms.Compose(trans)  # 组合所有变换
    mnist_train = torchvision.datasets.FashionMNIST(root=dataset_path, train=True, download=False, transform=transform)  # 加载训练集
    mnist_test = torchvision.datasets.FashionMNIST(root=dataset_path, train=False, download=False, transform=transform)  # 加载测试集
    
    if sys.platform.startswith('win'):  # 如果是Windows系统
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:  # 如果是其他系统
        num_workers = 4  # 使用4个工作进程
    
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # 创建训练数据迭代器
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # 创建测试数据迭代器
    
    return train_iter, test_iter  # 返回训练和测试数据迭代器

# 评估模型准确率
def evaluate_accuracy(data_iter, net, device=None):  # 定义评估模型准确率的函数
    if device is None and isinstance(net, nn.Module):  # 如果没有指定设备且是PyTorch模型
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device  # 获取模型所在的设备
    acc_sum, n = 0.0, 0  # 初始化准确率总和和样本数
    with torch.no_grad():  # 在不计算梯度的上下文中执行
        for X, y in data_iter:  # 遍历数据迭代器
            if isinstance(net, nn.Module):  # 如果是PyTorch模型
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()  # 计算正确预测数
                net.train()  # 改回训练模式
            else:  # 自定义的模型
                if('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()  # 计算正确预测数
                else:  # 如果没有is_training参数
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()  # 计算正确预测数
            n += y.shape[0]  # 累加样本数
    return acc_sum / n  # 返回准确率

# 训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):  # 定义训练模型的函数
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
                # 自定义的sgd函数
                for param in params:  # 遍历所有参数
                    param.data -= lr * param.grad / batch_size  # 使用梯度更新参数
            else:  # 如果使用了优化器
                optimizer.step()  # 使用优化器更新参数
            
            train_l_sum += l.item()  # 累加训练损失
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()  # 累加正确预测数
            n += y.shape[0]  # 累加样本数
        test_acc = evaluate_accuracy(test_iter, net)  # 评估测试集准确率
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %  # 打印训练结果
              (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))  # 格式化输出

# 初始化权重
def init_weights(m):  # 定义初始化权重的函数
    if type(m) == nn.Linear:  # 如果是线性层
        nn.init.normal_(m.weight, std=0.01)  # 使用正态分布初始化权重

# 主程序
if __name__ == '__main__':  # 当脚本作为主程序运行时执行
    batch_size, lr, num_epochs = 256, 0.1, 10  # 设置批次大小、学习率和训练轮数

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))  # 定义多层感知机模型（展平层+线性层+ReLU+线性层）
    net.apply(init_weights)  # 应用权重初始化

    loss = nn.CrossEntropyLoss(reduction='none')  # 定义交叉熵损失函数
    trainer = torch.optim.SGD(net.parameters(), lr=lr)  # 定义随机梯度下降优化器
    
    train_iter, test_iter = load_data_fashion_mnist(batch_size)  # 加载数据
    train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer=trainer)  # 训练模型