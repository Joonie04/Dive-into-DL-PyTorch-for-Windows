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

**参数说明**：

- `num_inputs = 784`：输入特征数
  - Fashion-MNIST 图像大小为 28×28 像素
  - 展平后为 784 个特征

- `num_outputs = 10`：输出类别数
  - Fashion-MNIST 有 10 个类别（t-shirt, trouser, pullover 等）

- `W`：权重矩阵，形状 `(784, 10)`
  - 每一列对应一个类别的权重
  - 使用小随机值初始化，避免对称性问题

- `b`：偏置向量，形状 `(10,)`
  - 每个元素对应一个类别的偏置
  - 初始化为 0

### 3.6.2 定义模型

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

**Softmax 函数详解**：

**数学公式**：
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{C} e^{x_j}}$$

**计算步骤**：
1. **计算指数**：`X_exp = torch.exp(X)`
   - 将任意实数转换为正数
   - 例如：`[2.0, 1.0, 0.1]` → `[7.389, 2.718, 1.105]`

2. **计算总和**：`partition = X_exp.sum(1, keepdim=True)`
   - 对每行求和（每个样本的所有类别）
   - `keepdim=True` 保持维度，便于广播

3. **归一化**：`X_exp / partition`
   - 除以总和，得到概率分布
   - 所有概率之和为 1

**为什么 Softmax 输出是概率？**

| 性质 | 说明 |
|------|------|
| 非负性 | $e^{x_i} > 0$，所以 $\text{softmax}(x_i) > 0$ |
| 归一化 | $\sum_{i=1}^{C} \text{softmax}(x_i) = 1$ |
| 范围 | 所有值在 $[0, 1]$ 之间 |

**网络结构**：
```
输入 X (batch_size, 784)
    ↓
线性变换: X @ W + b
    ↓
输出 (batch_size, 10)  # logits
    ↓
Softmax 归一化
    ↓
概率分布 (batch_size, 10)
```

### 3.6.3 定义损失函数

```python
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
```

**交叉熵损失详解**：

**数学公式**：
$$H(y, \hat{y}) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

**为什么使用交叉熵？**

**1. 适合多分类问题**：
- 线性回归：预测连续值 → 使用均方误差（MSE）
- 二分类问题：预测是/否 → 使用二元交叉熵
- **多分类问题**：预测多个类别 → 使用交叉熵

**2. 只关注真实类别**：

假设有 3 个类别，真实标签是第 0 类（猫）：

```python
y_hat = [0.7, 0.2, 0.1]  # 预测概率
y = 0                     # 真实标签

# 交叉熵只计算真实类别的损失
loss = -log(0.7) = 0.357
```

**3. 对比均方误差（MSE）**：

| 损失函数 | 适用场景 | 特点 |
|---------|---------|------|
| MSE | 回归问题 | 对所有维度计算误差 |
| 交叉熵 | 多分类 | 只关注真实类别的预测概率 |

**交叉熵的优势**：
- 梯度下降时收敛更快
- 数值稳定性更好
- 适合概率分布的输出

**代码解析**：
```python
y_hat[range(len(y_hat)), y]
```
- `range(len(y_hat))`：生成 `[0, 1, 2, ..., batch_size-1]`
- `y`：真实标签，例如 `[0, 2, 1, ...]`
- `y_hat[range(len(y_hat)), y]`：提取每个样本真实类别的预测概率
  - 第 0 个样本的第 0 类的概率
  - 第 1 个样本的第 2 类的概率
  - 第 2 个样本的第 1 类的概率
  - ...

### 3.6.4 定义准确率

```python
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
```

**准确率计算步骤**：

1. **获取预测类别**：`y_hat.argmax(axis=1)`
   - `y_hat` 形状：`(batch_size, 10)`
   - `argmax(axis=1)`：返回每行最大值的索引（预测类别）
   - 例如：`[[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]` → `[0, 2]`

2. **比较预测和真实标签**：`y_hat.type(y.dtype) == y`
   - 返回布尔张量，True 表示预测正确

3. **计算正确数量**：`cmp.type(y.dtype).sum()`
   - 将布尔值转换为整数（True=1, False=0）
   - 求和得到正确预测的数量

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
```

**Accumulator 类说明**：
- 用于在训练过程中累加多个指标
- 例如：累加损失、正确预测数、样本数
- `add(*args)`：累加多个值到对应的变量

```python
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
```

**evaluate_accuracy 函数说明**：

**为什么要设置评估模式？**

| 行为 | 训练模式 | 评估模式 |
|------|-----------|-----------|
| Dropout | 随机丢弃神经元 | 不丢弃（使用全部神经元） |
| Batch Normalization | 使用当前批次的统计量 | 使用训练时的统计量 |
| 梯度计算 | 需要计算 | 不需要计算 |

**为什么使用 `torch.no_grad()`？**
- 评估时不需要计算梯度
- 节省内存和计算资源
- 避免意外修改梯度

```python
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
```

**train_epoch_ch3 函数说明**：

**训练步骤**：

1. **设置训练模式**：`net.train()`
   - 启用 Dropout、Batch Normalization 等训练特有行为

2. **前向传播**：`y_hat = net(X)`
   - 计算预测值

3. **计算损失**：`l = loss(y_hat, y)`
   - 计算交叉熵损失

4. **反向传播**：`l.sum().backward()`
   - 计算梯度

5. **更新参数**：`updater(X.shape[0])`
   - 使用梯度下降更新参数

6. **累加指标**：`metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())`
   - 累加损失、正确预测数、样本数

**为什么要累加指标？**

每个 epoch 有多个批次，需要统计所有批次的总损失、总正确数、总样本数，最后计算平均值：

```python
# 平均损失 = 总损失 / 总样本数
# 平均准确率 = 总正确数 / 总样本数
return metric[0] / metric[2], metric[1] / metric[2]
```

**训练模式 vs 评估模式对比**：

| 模式 | 代码 | 作用 |
|------|------|------|
| 训练模式 | `net.train()` | 启用 Dropout、Batch Norm 等训练特有行为 |
| 评估模式 | `net.eval()` | 关闭训练特有行为，使用训练时的统计量 |

**完整训练流程**：

```
每个 epoch：
  ├── 遍历所有批次
  │   ├── 前向传播：计算预测值
  │   ├── 计算损失：交叉熵
  │   ├── 反向传播：计算梯度
  │   ├── 更新参数：梯度下降
  │   └── 累加指标：损失、正确数、样本数
  ├── 计算平均损失和准确率
  └── 评估测试集准确率
```

```python
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

**代码说明**：

**1. 定义模型**：
```python
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
```

**网络结构**：

| 层 | 作用 | 输入形状 | 输出形状 |
|---|------|---------|---------|
| `nn.Flatten()` | 展平图像 | (batch, 28, 28) | (batch, 784) |
| `nn.Linear(784, 10)` | 线性变换 | (batch, 784) | (batch, 10) |

**重要说明**：
- 网络输出的是 **logits**（未归一化的分数），不是概率
- logits 可以是任意实数，没有经过 softmax 归一化

**2. 初始化模型参数**：
```python
net.apply(init_weights)
```
- `net.apply(init_weights)`：递归地对所有层应用初始化函数
- `init_weights`：对线性层的权重使用正态分布初始化

**3. 定义损失函数**：
```python
loss = nn.CrossEntropyLoss(reduction='none')
```

**关键问题：Softmax 在哪里？**

**答案**：`nn.CrossEntropyLoss` 内部已经包含了 softmax 操作！

**nn.CrossEntropyLoss 的内部实现**：
```python
nn.CrossEntropyLoss() = nn.LogSoftmax() + nn.NLLLoss()
```

**计算流程**：
```
输入 X
    ↓
nn.Flatten()  # 展平：(batch, 28, 28) → (batch, 784)
    ↓
nn.Linear(784, 10)  # 线性变换：(batch, 784) → (batch, 10)
    ↓
输出 logits（未归一化的分数）
    ↓
nn.CrossEntropyLoss 内部：
    ├── LogSoftmax：log(softmax(logits))
    └── NLLLoss：计算负对数似然
```

**4. 两种实现的对比**：

| 实现 | 网络输出 | 损失函数 | Softmax 在哪里？ |
|------|---------|---------|-----------------|
| 3_6.py（从零开始） | 概率 | 手动交叉熵 | 网络中手动定义 |
| 3_7.py（简洁实现） | logits | nn.CrossEntropyLoss | 损失函数内部 |

**从零开始实现（3_6.py）**：
```python
# 手动定义 softmax
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

# 网络输出概率
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[1])), W) + b)

# 手动定义交叉熵
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
```

**简洁实现（3_7.py）**：
```python
# 网络输出 logits（未归一化的分数）
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

# CrossEntropyLoss 内部包含 softmax
loss = nn.CrossEntropyLoss(reduction='none')
```

**5. 为什么这样设计？**

**数值稳定性**：

手动实现可能出现数值问题：
```python
# 可能出现数值不稳定
softmax = exp(x) / sum(exp(x))
log_softmax = log(softmax)  # 可能出现 log(0) 的问题
```

PyTorch 内部使用数值稳定的实现：
```python
# 数值稳定的实现
log_softmax = x - log(sum(exp(x)))
# 避免了 exp 和 log 的数值问题
```

**6. 如何获取概率？**

如果需要概率，可以手动添加 softmax：

```python
# 方法1：训练时不加 softmax，推理时加
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
logits = net(X)
probs = torch.softmax(logits, dim=1)  # 手动计算概率

# 方法2：在模型中包含 softmax（不推荐用于训练）
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10),
    nn.Softmax(dim=1)  # 不推荐，会导致数值不稳定
)
```

**7. 训练流程**：

```python
# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
for epoch in range(num_epochs):
    for X, y in train_iter:
        # 前向传播：输出 logits
        logits = net(X)
        # 计算损失：内部包含 softmax
        l = loss(logits, y)
        # 清零梯度
        trainer.zero_grad()
        # 反向传播
        l.mean().backward()
        # 更新参数
        trainer.step()
```

**总结**：

| 特性 | 说明 |
|------|------|
| 网络输出 | logits（未归一化的分数） |
| Softmax 位置 | 在 `nn.CrossEntropyLoss` 内部 |
| 优势 | 数值稳定性更好 |
| 推荐做法 | 训练时使用 logits + CrossEntropyLoss，推理时手动计算 softmax |

## 3.8 总结

1. **线性回归**：用于预测连续值，使用均方误差作为损失函数，通过梯度下降进行优化。
2. **Softmax回归**：用于多分类问题，将线性输出转换为概率分布，使用交叉熵作为损失函数。
3. **Fashion-MNIST数据集**：用于图像分类任务的常用数据集。
4. **PyTorch API**：提供了简洁的方式来实现线性模型，包括数据加载、模型定义、损失函数和优化算法。
5. **梯度下降**：一种常用的优化算法，通过迭代更新参数来最小化损失函数。

通过本章的学习，我们掌握了线性神经网络的基本概念和实现方法，为后续学习更复杂的深度学习模型打下了基础。