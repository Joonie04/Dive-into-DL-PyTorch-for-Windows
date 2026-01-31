import random  # 导入随机数模块
import torch  # 导入PyTorch库

def synthetic_data(w, b, num_examples):  # 定义生成合成数据的函数
    """生成y=Xw+b+噪声"""  # 函数文档字符串
    X = torch.normal(0, 1, (num_examples, len(w)))  # 生成均值为0、标准差为1的正态分布特征矩阵
    y = torch.matmul(X, w) + b  # 计算线性回归的输出 y = Xw + b
    y += torch.normal(0, 0.01, y.shape)  # 添加均值为0、标准差为0.01的高斯噪声
    return X, y.reshape((-1, 1))  # 返回特征矩阵X和重塑后的标签y

def deta_iter(batch_size, features, labels):  # 定义数据迭代器函数
    """随机读取小批量"""  # 函数文档字符串
    num_examples = len(features)  # 获取样本数量
    indices = list(range(num_examples))  # 创建样本索引列表
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)  # 随机打乱索引顺序
    for i in range(0, num_examples, batch_size):  # 按批次大小遍历数据
        batch_indices = torch.tensor(  # 将当前批次的索引转换为张量
            indices[i: min(i + batch_size, num_examples)])  # 获取当前批次的索引
        yield features[batch_indices], labels[batch_indices]  # 生成当前批次的特征和标签

def linreg(X, w, b):  # 定义线性回归模型函数
    """线性回归模型"""  # 函数文档字符串
    return torch.matmul(X, w) + b  # 计算线性回归预测值 y_hat = Xw + b

def squared_loss(y_hat, y):  # 定义均方损失函数
    """均方损失"""  # 函数文档字符串
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # 计算均方损失 (y_hat - y)^2 / 2

def sgd(params, lr):  # 定义小批量随机梯度下降函数
    """小批量随机梯度下降"""  # 函数文档字符串
    with torch.no_grad():  # 在不计算梯度的上下文中执行
        for param in params:  # 遍历所有参数
            param -= lr * param.grad / batch_size  # 使用梯度更新参数
            param.grad.zero_()  # 将参数梯度清零

if __name__ == '__main__':  # 当脚本作为主程序运行时执行
    lr = 0.03  # 设置学习率
    num_epochs = 3  # 设置训练轮数
    net = linreg  # 设置网络模型为线性回归
    loss = squared_loss  # 设置损失函数为均方损失
    batch_size = 10  # 设置批次大小
    # 真正的参数用于生成数据
    true_w = torch.tensor([2.0, -3.4])  # 设置真实的权重参数
    true_b = 4.2  # 设置真实的偏置参数
    features, labels = synthetic_data(true_w, true_b, 1000)  # 生成1000个样本的合成数据
    
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # 初始化权重参数，需要计算梯度
    b = torch.zeros(1, requires_grad=True)  # 初始化偏置参数，需要计算梯度

    print(f'真实参数: w={true_w}, b={true_b}')  # 打印真实参数
    print(f'初始预测参数: w={w.squeeze()}, b={b.squeeze()}')  # 打印初始预测参数

    for epoch in range(num_epochs):  # 遍历每一轮训练
        for X, y in deta_iter(batch_size, features, labels):  # 遍历每个小批量数据
            y_hat = net(X, w, b)  # 计算预测值
            l = loss(y_hat, y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            # l.sum().backward(retain_graph=True)
            l.sum().backward()  # 计算损失关于参数的梯度
            sgd([w, b], lr)  # 使用参数的梯度更新参数
        with torch.no_grad():  # 在不计算梯度的上下文中执行
            train_l = loss(net(features, w, b), labels)  # 计算训练损失
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')  # 打印当前轮次和损失
            print(f'  参数w: {w.squeeze()}, 估计误差: {(w.squeeze() - true_w).abs().mean().item():.6f}')  # 打印权重参数和估计误差
            print(f'  参数b: {b.squeeze()}, 估计误差: {(b.squeeze() - true_b).abs().item():.6f}')  # 打印偏置参数和估计误差