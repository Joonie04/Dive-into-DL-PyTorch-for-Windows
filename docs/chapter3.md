# 第三章：线性神经网络

## 3.1 线性回归

### 3.1.1 模型定义
线性回归是一种监督学习算法，用于预测连续值。其数学表达式为：

$$y = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

其中：
- $y$ 是预测值
- $x_1, x_2, ..., x_n$ 是输入特征
- $w_1, w_2, ..., w_n$ 是权重
- $b$ 是偏置项

### 3.1.2 损失函数
线性回归使用均方误差（MSE）作为损失函数：

$$L(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

其中：
- $m$ 是样本数量
- $y_i$ 是真实值
- $\hat{y}_i$ 是预测值

### 3.1.3 梯度下降
梯度下降是一种优化算法，用于最小化损失函数。其更新规则为：

$$w = w - \alpha \frac{\partial L}{\partial w}$$
$$b = b - \alpha \frac{\partial L}{\partial b}$$

其中 $\alpha$ 是学习率。

## 3.2 线性回归的实现

### 3.2.1 生成数据集
使用随机数据生成器创建线性回归数据集：

```python
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
```

### 3.2.2 读取数据集
实现数据迭代器，用于批量读取数据：

```python
def data_iter(batch_size, features, labels):
    """随机读取小批量"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

### 3.2.3 定义模型
实现线性回归模型：

```python
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b
```

### 3.2.4 定义损失函数
实现均方损失函数：

```python
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

### 3.2.5 定义优化算法
实现小批量随机梯度下降：

```python
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

### 3.2.6 训练模型
训练线性回归模型：

```python
# 初始化参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练模型
lr = 0.03
num_epochs = 3
batch_size = 10
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = linreg(X, w, b)
        l = squared_loss(y_hat, y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

## 3.3 线性回归的简洁实现

### 3.3.1 使用PyTorch的API
使用PyTorch的内置API实现线性回归：

```python
# 导入必要的库
import torch
from torch.utils import data
from torch import nn

# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 读取数据集
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

## 3.4 Softmax回归

### 3.4.1 模型定义
Softmax回归是一种用于多分类问题的线性模型。其数学表达式为：

$$\hat{y} = softmax(XW + b)$$

其中 $softmax$ 函数定义为：

$$softmax(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}$$

### 3.4.2 损失函数
Softmax回归使用交叉熵损失函数：

$$L(y, \hat{y}) = -\sum_{i=1}^{k} y_i \log(\hat{y}_i)$$

### 3.4.3 实现Softmax回归

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
```

## 3.5 图像分类数据集

### 3.5.1 Fashion-MNIST数据集
Fashion-MNIST是一个服装分类数据集，包含10个类别的60,000个训练样本和10,000个测试样本。

### 3.5.2 读取数据集

```python
trans = transforms.ToTensor()
train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=trans, download=False)
test_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=trans, download=False)

train_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

### 3.5.3 可视化数据集

```python
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, title=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if title:
            ax.set_title(title[i])
    return axes
```

## 3.6  Softmax回归的实现

### 3.6.1 初始化模型参数

```python
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

### 3.6.2 定义模型

```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

### 3.6.3 定义损失函数

```python
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
```

### 3.6.4 定义准确率

```python
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
```

### 3.6.5 训练模型

```python
# 定义Accumulator类
class Accumulator:
    '''
    用于累加多个变量的类
    '''
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# 定义evaluate_accuracy函数
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 定义train_epoch_ch3函数
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            l = l.item()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

# 定义Animator类
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        # SVG显示已在文件开头设置
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        self.epoch_count = 0

    def add(self, x, y):
        # 向列表添加数据
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in self.fmts]
        if not self.Y:
            self.Y = [[] for _ in self.fmts]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        
        self.epoch_count += 1
        
        # 每个epoch结束或者训练完成时显示图像
        if self.epoch_count % 1 == 0:  # 每次添加都显示，更新图像
            plt.show(block=False)
            plt.pause(0.001)

# 定义set_axes函数
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

# 定义train_ch3函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
```

## 3.7 Softmax回归的简洁实现

### 3.7.1 使用PyTorch的API

```python
# 定义初始化权重函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# 定义模型
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

# 初始化模型参数
net.apply(init_weights)

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 3.8 总结

1. **线性回归**：用于预测连续值，使用均方误差作为损失函数，通过梯度下降进行优化。
2. **Softmax回归**：用于多分类问题，将线性输出转换为概率分布，使用交叉熵作为损失函数。
3. **Fashion-MNIST数据集**：用于图像分类任务的常用数据集。
4. **PyTorch API**：提供了简洁的方式来实现线性模型，包括数据加载、模型定义、损失函数和优化算法。
5. **梯度下降**：一种常用的优化算法，通过迭代更新参数来最小化损失函数。

通过本章的学习，我们掌握了线性神经网络的基本概念和实现方法，为后续学习更复杂的深度学习模型打下了基础。