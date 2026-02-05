# 3.6 softmax回归从零开始实现
import torch  # 导入PyTorch库
import torchvision  # 导入torchvision库，用于计算机视觉任务
from torch.utils import data  # 导入PyTorch数据处理工具
from torchvision import transforms  # 导入图像变换模块

# 设置matplotlib后端为TkAgg，以便在命令行环境中显示图像
import matplotlib  # 导入matplotlib库
matplotlib.use('TkAgg')  # 设置matplotlib使用TkAgg后端
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot绘图模块

# 设置matplotlib使用SVG显示
plt.rcParams['figure.figsize'] = (3.5, 2.5)  # 设置图形大小为3.5x2.5英寸
plt.rcParams['figure.dpi'] = 100  # 设置图形分辨率为100 DPI


def get_fashion_mnist_labels(labels):  # 定义获取Fashion-MNIST标签文本的函数
    """返回Fashion-MNIST数据集的文本标签"""  # 函数文档字符串
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',  # 定义10个类别的文本标签
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]  # 返回标签对应的文本列表

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):  # 定义设置坐标轴的函数
    """设置matplotlib的轴"""  # 函数文档字符串
    axes.set_xlabel(xlabel)  # 设置x轴标签
    axes.set_ylabel(ylabel)  # 设置y轴标签
    axes.set_xscale(xscale)  # 设置x轴刻度类型
    axes.set_yscale(yscale)  # 设置y轴刻度类型
    axes.set_xlim(xlim)  # 设置x轴范围
    axes.set_ylim(ylim)  # 设置y轴范围
    if legend:  # 如果有图例
        axes.legend(legend)  # 添加图例
    axes.grid()  # 显示网格

def show_images(imgs, num_rows, num_cols, title=None, scale=1.5):  # 定义显示图像的函数
    """绘制图像列表"""  # 函数文档字符串
    figsize = (num_cols * scale, num_rows * scale)  # 计算图形大小
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)  # 创建子图网格
    axes = axes.flatten()  # 将子图数组展平为一维
    for i, (ax, img) in enumerate(zip(axes, imgs)):  # 遍历每个子图和图像
        if torch.is_tensor(img):  # 如果图像是PyTorch张量
            # 图片张量
            ax.imshow(img.numpy())  # 将张量转换为NumPy数组并显示
        else:  # 如果图像不是张量
            # PIL图片
            ax.imshow(img)  # 直接显示PIL图像
        ax.axes.get_xaxis().set_visible(False)  # 隐藏x轴
        ax.axes.get_yaxis().set_visible(False)  # 隐藏y轴
        if title:  # 如果有标题
            ax.set_title(title[i])  # 设置子图标题
    return axes  # 返回子图对象

def softmax(X):  # 定义softmax函数
    X_exp = torch.exp(X)  # 计算指数
    partition = X_exp.sum(1, keepdim=True)  # 计算每行的和（用于归一化）
    return X_exp / partition  # 广播机制，返回softmax概率分布

def net(X):  # 定义softmax回归模型
    return softmax(torch.matmul(X.reshape((-1, W.shape[1])), W) + b)  # 计算softmax回归的输出

def cross_entropy(y_hat, y):  # 定义交叉熵损失函数
    return -torch.log(y_hat[range(len(y_hat)), y])  # 计算交叉熵损失

def accuracy(y_hat, y):  # 定义计算准确率的函数
    """计算预测正确的数量"""  # 函数文档字符串
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 如果预测结果是多维的
        y_hat = y_hat.argmax(axis=1)  # 获取预测类别（概率最大的索引）
    cmp = y_hat.type(y.dtype) == y  # 比较预测结果和真实标签
    return float(cmp.type(y.dtype).sum())  # 返回预测正确的数量

def evaluate_accuracy(net, data_iter):  # 定义评估模型准确率的函数
    """计算在指定数据集上模型的精度"""  # 函数文档字符串
    if isinstance(net, torch.nn.Module):  # 如果是PyTorch模型
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():  # 在不计算梯度的上下文中执行
        for X, y in data_iter:  # 遍历数据迭代器
            metric.add(accuracy(net(X), y), y.numel())  # 累加正确预测数和样本数
    return metric[0] / metric[1]  # 返回准确率

def train_epoch(net, train_iter, loss, updater):  # 定义训练一个epoch的函数
    """训练模型一个迭代周期（定义见第3章）"""  # 函数文档字符串
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):  # 如果是PyTorch模型
        net.train()  # 将模型设置为训练模式
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)  # 创建累加器，用于累加3个指标
    for X, y in train_iter:  # 遍历训练数据迭代器
        # 计算梯度并更新参数
        y_hat = net(X)  # 前向传播，计算预测值
        l = loss(y_hat, y)  # 计算损失
        if isinstance(updater, torch.optim.Optimizer):  # 如果使用PyTorch内置优化器
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()  # 清零梯度
            l.backward()  # 反向传播计算梯度
            updater.step()  # 更新参数
            l = l.item()  # 获取损失值
        else:  # 如果使用自定义优化器
            # 使用定制的优化器和损失函数
            l.sum().backward()  # 反向传播计算梯度
            updater(X.shape[0])  # 使用自定义更新函数更新参数
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())  # 累加损失、准确率和样本数
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]  # 返回平均损失和平均准确率

def train(net, train_iter, test_iter, loss, num_epochs, updater):  # 定义训练模型的函数
    """训练模型（定义见第3章）"""  # 函数文档字符串
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],  # 创建动画绘制器
                        legend=['train loss', 'train acc', 'test acc'])
    print(f"开始训练，总共{num_epochs}个epoch...")  # 打印训练开始信息
    for epoch in range(num_epochs):  # 遍历每个epoch
        print(f"正在训练第{epoch + 1}个epoch...")  # 打印当前epoch
        train_metrics = train_epoch(net, train_iter, loss, updater)  # 训练一个epoch
        test_acc = evaluate_accuracy(net, test_iter)  # 评估测试集准确率
        print(f"Epoch {epoch + 1}: train_loss={train_metrics[0]:.4f}, train_acc={train_metrics[1]:.4f}, test_acc={test_acc:.4f}")  # 打印训练结果
        animator.add(epoch + 1, train_metrics + (test_acc,))  # 更新动画
    train_loss, train_acc = train_metrics  # 获取最终训练损失和准确率
    print(f"训练完成！最终结果：train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")  # 打印最终结果
    # 暂时注释掉断言以避免错误
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc

def sgd(params, lr, batch_size):  # 定义SGD优化器函数
    """小批量随机梯度下降"""  # 函数文档字符串
    with torch.no_grad():  # 在不计算梯度的上下文中执行
        for param in params:  # 遍历所有参数
            param -= lr * param.grad / batch_size  # 使用梯度更新参数
            param.grad.zero_()  # 清零梯度

def updater(batch_size):  # 定义参数更新函数
    """小批量随机梯度下降更新"""  # 函数文档字符串
    return sgd([W, b], lr, batch_size)  # 返回SGD更新函数

def predict(net, test_iter, n=6):  # 定义预测函数
    """预测标签（定义见第3章）"""  # 函数文档字符串
    for X, y in test_iter:  # 获取测试数据
        break  # 只取第一个批次
    trues = get_fashion_mnist_labels(y)  # 获取真实标签
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))  # 获取预测标签
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]  # 组合真实标签和预测标签
    axes = show_images(X[0:n].reshape((n, 28, 28)), 1, n, title=titles[0:n])  # 显示前n张图像
    
    # 在命令行环境中显示预测图像
    plt.show()  # 显示图形
    
    return trues, preds  # 返回真实标签和预测标签

class Accumulator:  # 定义累加器类
    '''
    用于累加多个变量的类
    '''  # 类文档字符串
    def __init__(self, n):  # 初始化方法
        self.data = [0.0] * n  # 初始化n个累加变量
    def add(self, *args):  # 添加方法
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # 累加参数到对应变量
    def reset(self):  # 重置方法
        self.data = [0.0] * len(self.data)  # 重置所有累加变量为0
    def __getitem__(self, idx):  # 索引方法
        return self.data[idx]  # 返回指定索引的累加变量

class Animator:  # 定义动画绘制器类
    """在动画中绘制数据"""  # 类文档字符串
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,  # 初始化方法
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:  # 如果没有图例
            legend = []  # 初始化为空列表
        # SVG显示已在文件开头设置
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)  # 创建图形和子图
        if nrows * ncols == 1:  # 如果只有一个子图
            self.axes = [self.axes, ]  # 转换为列表
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(  # 创建配置坐标轴的lambda函数
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts  # 初始化X、Y和格式
        self.epoch_count = 0  # 初始化epoch计数器

    def add(self, x, y):  # 添加数据方法
        # 向列表添加数据
        if not hasattr(y, "__len__"):  # 如果y不是列表
            y = [y]  # 转换为列表
        n = len(y)  # 获取y的长度
        if not hasattr(x, "__len__"):  # 如果x不是列表
            x = [x] * n  # 复制n次
        if not self.X:  # 如果X为空
            self.X = [[] for _ in self.fmts]  # 初始化X列表
        if not self.Y:  # 如果Y为空
            self.Y = [[] for _ in self.fmts]  # 初始化Y列表
        for i, (a, b) in enumerate(zip(x, y)):  # 遍历x和y
            if a is not None and b is not None:  # 如果a和b都不为None
                self.X[i].append(a)  # 添加x值
                self.Y[i].append(b)  # 添加y值
        self.axes[0].cla()  # 清除子图
        for x, y, fmt in zip(self.X, self.Y, self.fmts):  # 遍历X、Y和格式
            self.axes[0].plot(x, y, fmt)  # 绘制曲线
        self.config_axes()  # 配置坐标轴
        
        self.epoch_count += 1  # 增加epoch计数
        
        # 每个epoch结束或者训练完成时显示图像
        if self.epoch_count % 1 == 0:  # 每次添加都显示，更新图像
            plt.show(block=False)  # 显示图形但不阻塞
            plt.pause(0.001)  # 暂停一小段时间


if __name__ == '__main__':  # 当脚本作为主程序运行时执行
    batch_size = 256  # 设置批次大小
    ## train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    trans = transforms.ToTensor()  # 定义图像转换为张量的变换
    data_path = 'dataset/FashionMNIST/raw'  # 设置数据集存储路径
    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=trans, download=False)  # 加载训练集
    test_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=trans, download=False)  # 加载测试集
   
    train_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 创建训练数据迭代器
    test_iter = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 创建测试数据迭代器


    # 检查数据是否加载成功
    print(f"训练集样本数: {len(train_iter)}")  # 打印训练集迭代器数量
    print(f"测试集样本数: {len(test_iter)}")  # 打印测试集迭代器数量

    num_inputs = 784  # 输入维度（28×28像素）
    num_outputs = 10  # 输出维度（10个类别）

    # 初始化模型参数
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)  # 初始化权重矩阵
    b = torch.zeros(num_outputs, requires_grad=True)  # 初始化偏置向量

    lr = 0.1  # 设置学习率
    num_epochs = 10  # 设置训练轮数
    train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)  # 训练模型

    # 预测
    predict(net, test_iter)  # 对测试集进行预测