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

**代码说明**：
- `yield`：使用生成器按需生成数据，节省内存
- `random.shuffle(indices)`：随机打乱索引，确保每个 epoch 的数据顺序不同
- `range(0, num_examples, batch_size)`：从 0 开始，每次增加 batch_size
- `min(i + batch_size, num_examples)`：处理最后一个批次可能不足 batch_size 的情况
- `features[batch_indices]`：根据索引批量提取特征和标签

**执行流程**：
1. 创建样本索引列表并随机打乱
2. 按批次大小遍历索引
3. 每次取 batch_size 个索引（最后一个批次可能不足）
4. 根据索引从 features 和 labels 中提取数据
5. 通过 yield 逐批返回数据

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

**代码说明**：

- `torch.no_grad()`：禁用梯度计算的上下文管理器
  - 参数更新时不需要计算梯度，因为梯度已经在 `backward()` 时计算完成
  - 禁用梯度计算可以节省内存和计算资源
  - 避免构建不必要的计算图

- `param -= lr * param.grad / batch_size`：参数更新公式
  - `lr`：学习率，控制更新步长
  - `param.grad`：参数的梯度（通过反向传播计算得到）
  - `-=`：减法赋值，沿梯度的**反方向**更新参数
    - 为什么用 `-=` 而不是 `+=`：
      - 梯度指向损失函数**增长最快**的方向
      - 我们要**最小化**损失函数，所以要沿梯度的**反方向**移动
      - 如果用 `+=`，参数会沿梯度方向移动，损失会越来越大（梯度上升）
      - 举例：假设 $L(w) = w^2$，当前 $w=2$，梯度 $\frac{\partial L}{\partial w}=4$
        - `-=`: $w = 2 - 0.1 \times 4 = 1.6$，损失从 4 降到 2.56（正确）
        - `+=`: $w = 2 + 0.1 \times 4 = 2.4$，损失从 4 增加到 5.76（错误）
  - `/ batch_size`：除以批次大小，使用**平均梯度**而不是梯度之和
  - 为什么要除以 batch_size：
    - `loss(y_hat, y)` 返回每个样本的损失，形状为 `(batch_size, 1)`
    - `l.sum().backward()` 对损失求和后计算梯度
    - 因此 `param.grad` 是整个批次的梯度**之和**
    - 除以 `batch_size` 得到平均梯度，保持学习率的一致性
    - 如果不除，当 `batch_size` 变化时需要相应调整学习率

- `param.grad.zero_()`：清零梯度
  - PyTorch 的梯度默认是**累积**的
  - 每次调用 `backward()` 时，梯度会累加到 `param.grad` 上
  - 必须在每次更新后手动清零，否则下次计算会使用错误的梯度值
  - 下划线 `_` 表示原地操作（in-place operation）

**数学原理**：
- 损失函数：$L = \sum_{i=1}^{batch} \frac{1}{2}(y^{(i)} - \hat{y}^{(i)})^2$
- 梯度：$\frac{\partial L}{\partial w} = \sum_{i=1}^{batch} (y^{(i)} - \hat{y}^{(i)})x^{(i)}$
- 参数更新：$w \leftarrow w - \frac{lr}{batch\_size} \cdot \frac{\partial L}{\partial w}$

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

**代码说明**：

**1. 参数初始化**：
```python
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```
- `w`：权重参数，从均值为0、标准差为0.01的正态分布中随机初始化
  - 小随机值初始化可以避免对称性问题
  - `requires_grad=True` 表示需要计算梯度
- `b`：偏置参数，初始化为0
  - `requires_grad=True` 表示需要计算梯度

**2. 训练超参数**：
```python
lr = 0.03          # 学习率，控制参数更新的步长
num_epochs = 3     # 训练轮数，遍历整个数据集的次数
batch_size = 10    # 批次大小，每次训练使用的样本数
```

**3. 训练循环**：
```python
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # 前向传播
        y_hat = linreg(X, w, b)
        # 计算损失
        l = squared_loss(y_hat, y)
        # 反向传播
        l.sum().backward()
        # 参数更新
        sgd([w, b], lr, batch_size)
```

**训练步骤详解**：

- **前向传播**：`y_hat = linreg(X, w, b)`
  - 输入：特征矩阵 X，权重 w，偏置 b
  - 计算：`y_hat = X @ w + b`
  - 输出：预测值 y_hat

- **计算损失**：`l = squared_loss(y_hat, y)`
  - 计算每个样本的均方损失：`(y_hat - y)^2 / 2`
  - 返回形状为 `(batch_size, 1)` 的损失张量

- **反向传播**：`l.sum().backward()`
  - `l.sum()`：对批次中所有样本的损失求和
  - `.backward()`：计算损失对参数的梯度
  - 梯度存储在 `w.grad` 和 `b.grad` 中

- **参数更新**：`sgd([w, b], lr, batch_size)`
  - 使用梯度下降更新参数
  - `w = w - lr * w.grad / batch_size`
  - `b = b - lr * b.grad / batch_size`
  - 清零梯度，为下一批次做准备

**4. 评估训练效果**：
```python
with torch.no_grad():
    train_l = squared_loss(linreg(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```
- `torch.no_grad()`：禁用梯度计算，节省资源
- 使用所有数据计算当前的平均损失
- 打印每个 epoch 的损失值，观察训练进度

**前向传播和反向传播的数学原理**：

**前向传播**：
```
z = X @ w + b           # 线性变换
y_hat = z               # 预测值
loss = (y_hat - y)^2 / 2  # 均方损失
L = sum(loss)           # 总损失
```

**反向传播（链式法则）**：
```
∂L/∂loss = 1
∂loss/∂y_hat = (y_hat - y)
∂y_hat/∂z = 1
∂z/∂w = X
∂z/∂b = 1

∂L/∂w = X^T @ (y_hat - y)
∂L/∂b = sum(y_hat - y)
```

**梯度计算说明**：
- `∂z/∂w = X`：雅可比矩阵，形状为 (n, m)
- `∂L/∂w = X^T @ (y_hat - y)`：在链式法则中需要转置雅可比矩阵
  - X^T 的形状为 (m, n)
  - (y_hat - y) 的形状为 (n, 1)
  - 结果形状为 (m, 1)，与 w 的形状一致

**参数更新**：
```
w = w - lr * ∂L/∂w / batch_size
b = b - lr * ∂L/∂b / batch_size
```

**训练过程**：
1. 每个 epoch 遍历整个数据集
2. 每个批次进行一次前向传播和反向传播
3. 使用梯度更新参数
4. 每个 epoch 结束后评估整体损失
5. 损失逐渐减小，模型参数逐渐接近真实值

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

**代码说明**：

**1. 数据生成和加载**：
```python
features, labels = synthetic_data(true_w, true_b, 1000)
data_iter = load_array((features, labels), batch_size)
```
- `synthetic_data`：生成合成数据集，包含特征和标签
- `load_array`：使用 PyTorch 的 DataLoader 创建数据迭代器
  - `batch_size`：每批次的样本数
  - `shuffle=is_train`：训练时打乱数据顺序

**2. 定义模型**：
```python
net = nn.Sequential(nn.Linear(2, 1))
```

**重要说明**：
- **只有一层**，不是两层！
- `nn.Sequential` 是一个容器，用于包装多个层，本身不是层
- `nn.Linear(2, 1)` 是一个线性层：
  - 输入维度：2（两个特征）
  - 输出维度：1（一个预测值）

**nn.Linear 的内部实现**：
```python
# nn.Linear 内部自动实现以下公式
y = X @ W^T + b
```
- `X`：输入特征矩阵，形状 (n, 2)
- `W`：权重矩阵，形状 (1, 2)
- `b`：偏置，形状 (1,)
- `y`：输出，形状 (n, 1)

**3. 初始化模型参数**：
```python
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```
- `net[0]`：访问第一个层（即 nn.Linear(2, 1)）
- `weight.data.normal_(0, 0.01)`：权重从均值为0、标准差为0.01的正态分布初始化
- `bias.data.fill_(0)`：偏置初始化为0

**4. 定义损失函数和优化器**：
```python
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```
- `nn.MSELoss()`：均方误差损失函数
- `torch.optim.SGD`：随机梯度下降优化器
  - `net.parameters()`：返回模型的所有可训练参数
  - `lr=0.03`：学习率

**5. 训练模型**：
```python
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
```

**训练步骤详解**：

- **前向传播**：`l = loss(net(X), y)`
  - `net(X)`：计算预测值 `y_hat = X @ W^T + b`
  - `loss(y_hat, y)`：计算均方误差

- **清零梯度**：`trainer.zero_grad()`
  - 清空所有参数的梯度
  - PyTorch 的梯度默认是累积的，必须手动清零

- **反向传播**：`l.backward()`
  - 计算损失对参数的梯度
  - 梯度存储在 `param.grad` 中

- **更新参数**：`trainer.step()`
  - **trainer.step() 的内部实现**：
    ```python
    for param in net.parameters():
        if param.grad is not None:
            # 执行：param = param - lr * param.grad
            param.data.add_(-lr, param.grad.data)
    ```
  - 使用梯度下降更新参数
  - `w = w - lr * w.grad`
  - `b = b - lr * b.grad`

**6. 评估训练效果**：
```python
l = loss(net(features), labels)
print(f'epoch {epoch + 1}, loss {l:f}')
```
- 使用所有数据计算当前损失
- 打印每个 epoch 的损失值

**获取学习到的参数**：
```python
w = net[0].weight.data
b = net[0].bias.data
```
- `net[0].weight.data`：学习到的权重
- `net[0].bias.data`：学习到的偏置

**多层网络的扩展**：

如果需要构建多层神经网络，可以在 `nn.Sequential` 中添加更多层：

```python
# 两层神经网络
net = nn.Sequential(
    nn.Linear(3, 3),  # 第一层：3维输入 → 3维隐藏层
    nn.Linear(3, 1)   # 第二层：3维隐藏层 → 1维输出
)

# 对应的数学公式
# y_hat = (X @ W1^T + b1) @ W2^T + b2
```

**多层网络的核心思想**：
- 每一层都是相同的模式：`输出 = 输入 @ 权重 + 偏置`
- 多层就是把上一层的输出作为下一层的输入
- 本质上就是不断嵌套 `@ W + b` 操作

**单层 vs 多层对比**：

| 模型 | 代码 | 公式 |
|------|------|------|
| 线性回归（单层） | `nn.Linear(2, 1)` | `y = X @ W + b` |
| 两层神经网络 | `nn.Linear(3, 3), nn.Linear(3, 1)` | `y = (X @ W1 + b1) @ W2 + b2` |
| 三层神经网络 | `nn.Linear(3, 5), nn.Linear(5, 3), nn.Linear(3, 1)` | `y = ((X @ W1 + b1) @ W2 + b2) @ W3 + b3` |

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