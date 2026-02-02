# 2 预备知识

本章介绍深度学习所需的预备知识，包括数据操作、数据预处理、线性代数、微积分、自动微分、概率和查阅文档等内容。

## 2.1 数据操作

张量（Tensor）是PyTorch中的核心数据结构，类似于NumPy数组，但可以在GPU上加速计算。

### 2.1.1 张量的创建与基本操作

**创建张量的常用方法：**

- `torch.arange(n)`：创建包含0到n-1的1D张量, 是按照步长为1的等差数列. 例如, `torch.arange(4)` 会创建一个张量 `tensor([0, 1, 2, 3])`.
- `torch.zeros(shape)`：创建全零张量. 如 `torch.zeros((1, 3))` 会创建一个张量 `tensor([[0., 0., 0.]])`, 形状为1x3的矩阵.
- `torch.ones(shape)`：创建全一张量. 如 `torch.ones((2, 3))` 会创建一个张量 `tensor([[1., 1., 1.], [1., 1., 1.]])`, 形状为2x3的矩阵.
- `torch.rand(shape)`：创建随机张量（元素在[0,1)区间）. 如 `torch.rand((2, 3))` 会创建一个张量 `tensor([[0.5444, 0.2268, 0.2627], [0.5349, 0.8403, 0.2761]])`, 形状为2x3的矩阵.
- `torch.tensor(data)`：从Python列表或NumPy数组创建张量. 如data = [1, 2, 3] 或 np.array([1, 2, 3]), 会创建一个张量 `tensor([1, 2, 3])`.

**张量的基本属性：**

**形状和维度：**
- `.shape`：张量的形状（返回torch.Size对象）
- `.size()`：获取张量的形状（与shape类似）
- `.ndim`：张量的维度数
- `.dim()`：获取张量的维度数（与ndim类似）
- `.numel()`：张量中元素的总数

**数据类型和设备：**
- `.dtype`：张量的数据类型（如torch.float32, torch.int64等）
    # 扩展说明：不同的数据类型在内存中占用不同的字节数，影响计算效率和内存占用。例如，torch.float32 占用4字节，torch.int64 占用8字节。
    # 常用数据类型
    类型 | 字节数 | 说明 | 范围
    --- | --- | --- | ---
    torch.int8 | 8 | 有符号8位整数 | [-128, 127]
    torch.uint8 | 8 | 无符号8位整数 | [0, 255]
    torch.int16 | 2 | 有符号16位整数 | [-2^15, 2^15-1]
    torch.int32 | 4 | 有符号32位整数 | [-2^31, 2^31-1]
    torch.int64 | 8 | 有符号64位整数 | [-2^63, 2^63-1]
    torch.float16 | 2 | 16位浮点数(半精度) | [-6.5e-5, 6.5e-5]
    torch.float32 | 4 | 32位浮点数(单精度) | [-3.4e38, 3.4e38]
    torch.float64 | 8 | 64位浮点数(双精度) | [-1.7e308, 1.7e308]
- `.device`：张量所在的设备（CPU或GPU）

**形状变换：**
- `.reshape(shape)`：重塑张量形状（不改变数据）
- `.view(shape)`：与`.reshape()`类似，但可能共享内存（仅在连续张量上有效）
- `.squeeze()`：移除所有长度为1的轴，返回一个新的张量，不改变原始张量。例如，`torch.tensor([[1, 2, 3]]).squeeze()` 会返回一个张量 `tensor([1, 2, 3])`
- `.squeeze(dim)`：移除指定维度上长度为1的轴
- `.unsqueeze(dim)`：在指定轴上添加长度为1的轴。例如，`torch.tensor([1, 2, 3]).unsqueeze(0)` 会返回一个张量 `tensor([[1, 2, 3]])`，形状为1x3的矩阵
- `.flatten()`：将张量展平为1D
- `.flatten(start_dim, end_dim)`：将指定维度范围展平

**维度操作：**
- `.transpose(dim0, dim1)`：交换两个维度
- `.permute(*dims)`：按照指定顺序重新排列维度
- `.t()`：矩阵转置（仅适用于2D张量，等同于`.transpose(0, 1)`）

**复制和扩展：**
- `.clone()`：创建张量的深拷贝
- `.repeat(*sizes)`：沿指定维度重复张量（复制数据）
- `.expand(*sizes)`：扩展张量（不复制数据，仅适用于维度为1的情况）

**内存和连续性：**
- `.is_contiguous()`：检查张量是否在内存中连续存储
- `.contiguous()`：返回内存连续的张量副本

**类型转换：**
- `.to(device)`：将张量移动到指定设备（CPU或GPU）
- `.to(dtype)`：转换张量的数据类型
- `.float()`：转换为float32类型
- `.int()`：转换为int32类型
- `.long()`：转换为int64类型
- `.double()`：转换为float64类型

**其他转换：**
- `.item()`：获取单元素张量的Python标量值
- `.tolist()`：将张量转换为Python列表
- `.numpy()`：将张量转换为NumPy数组

```python
x = torch.arange(12)  # 创建0-11的张量
X = x.reshape(3, 4)   # 重塑为3行4列
Z = torch.zeros((2, 3, 4))  # 创建2x3x4的全零张量
Y = torch.ones((3, 4))      # 创建3x4的全一张量
R = torch.rand((3, 4))      # 创建3x4的随机张量

# 基本属性
print("X.shape:", X.shape)      # torch.Size([3, 4])
print("X.size():", X.size())    # torch.Size([3, 4])
print("X.ndim:", X.ndim)        # 2
print("X.dim():", X.dim())      # 2
print("X.numel():", X.numel())  # 12
print("X.dtype:", X.dtype)       # torch.int64
print("X.device:", X.device)    # cpu

# 形状变换
X_squeezed = X.unsqueeze(0)     # 添加维度，形状变为1x3x4
print("X_squeezed.shape:", X_squeezed.shape)
X_flattened = X.flatten()       # 展平为1D
print("X_flattened.shape:", X_flattened.shape)

# 维度操作
X_transposed = X.transpose(0, 1)  # 转置，形状变为4x3
print("X_transposed.shape:", X_transposed.shape)
X_permuted = X.permute(1, 0)     # 与transpose(0, 1)相同
print("X_permuted.shape:", X_permuted.shape)

# 复制和扩展
X_cloned = X.clone()             # 深拷贝
X_repeated = X.repeat(2, 1)      # 沿维度0重复2次，维度1重复1次
print("X_repeated.shape:", X_repeated.shape)

# 类型转换
X_float = X.float()              # 转换为float32
print("X_float.dtype:", X_float.dtype)
```

### 2.1.2 张量的运算

**逐元素运算：**

**基本算术运算：**
- 加减乘除：`+`, `-`, `*`, `/`
- 整除：`//` 或 `torch.floor_divide(x, y)`（向下取整）
- 取余：`%` 或 `torch.remainder(x, y)`（取模）
- 乘方：`**` 或 `torch.pow(x, y)`
- 负数：`-x` 或 `torch.neg(x)`

**数学函数：**
- 指数：`torch.exp(x)`（e的x次方）
- 对数：`torch.log(x)`（自然对数）、`torch.log10(x)`（以10为底）、`torch.log2(x)`（以2为底）
- 平方根：`torch.sqrt(x)` 或 `x ** 0.5`
- 绝对值：`torch.abs(x)` 或 `x.abs()`
- 符号函数：`torch.sign(x)`（返回-1、0或1）
- 向上取整：`torch.ceil(x)`
- 向下取整：`torch.floor(x)`
- 四舍五入：`torch.round(x)`

**三角函数：**
- 正弦：`torch.sin(x)`
- 余弦：`torch.cos(x)`
- 正切：`torch.tan(x)`
- 反三角函数：`torch.asin(x)`, `torch.acos(x)`, `torch.atan(x)`

**比较运算：**
- 等于：`==` 或 `torch.eq(x, y)`
- 不等于：`!=` 或 `torch.ne(x, y)`
- 大于：`>` 或 `torch.gt(x, y)`
- 小于：`<` 或 `torch.lt(x, y)`
- 大于等于：`>=` 或 `torch.ge(x, y)`
- 小于等于：`<=` 或 `torch.le(x, y)`

**逻辑运算：**
- 逻辑与：`torch.logical_and(x, y)`
- 逻辑或：`torch.logical_or(x, y)`
- 逻辑非：`torch.logical_not(x)`

**聚合运算：**
- `.sum()`：求和
- `.mean()`：求平均值
- `.max()`：求最大值
- `.min()`：求最小值
- `.prod()`：求乘积
- `.std()`：求标准差
- `.var()`：求方差
- `.argmax()`：返回最大值的索引
- `.argmin()`：返回最小值的索引

**拼接运算：**
- `torch.cat((X, Y), dim=0)`：沿维度0（行）拼接
- `torch.cat((X, Y), dim=1)`：沿维度1（列）拼接
- `torch.stack((X, Y), dim=0)`：沿新维度堆叠

```python
x = torch.tensor([[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]])
y = torch.tensor([[2, 3, 2, 2], [2, 3, 2, 2], [2, 3, 2, 2]])

print(x + y)  # 逐元素相加
print(x - y)  # 逐元素相减
print(x * y)  # 逐元素相乘
print(x / y)  # 逐元素相除
print(x // y)  # 整除
print(x % y)  # 取余
print(x ** y) # 逐元素乘方
print(torch.exp(x))  # 逐元素计算指数
print(torch.log(x))  # 逐元素计算对数
print(torch.sqrt(x))  # 逐元素计算平方根
print(torch.abs(x))  # 逐元素计算绝对值
print(torch.ceil(x))  # 向上取整
print(torch.floor(x))  # 向下取整
print(torch.round(x))  # 四舍五入
print(torch.sin(x))  # 正弦函数
print(torch.cos(x))  # 余弦函数
print(x > y)  # 逐元素比较
print(x == y)  # 逐元素相等比较
print(x.max())  # 最大值
print(x.min())  # 最小值
print(x.mean())  # 平均值
print(x.std())  # 标准差
print(x.argmax())  # 最大值索引
print(x.argmin())  # 最小值索引
```

### 2.1.3 张量的广播机制

广播机制允许不同形状的张量进行运算，通过自动扩展维度使形状匹配。

**广播规则：**
1. 从最后一个维度开始比较
2. 维度相同或其中一个为1时可以广播
3. 缺失维度可以扩展

```python
x = torch.arange(3).reshape((3, 1))  # 3x1
y = torch.arange(2).reshape((1, 2))  # 1x2
print(x + y)  # 广播为3x2张量
```

### 2.1.4 张量的索引和切片

**索引和切片操作：**

- `X[-1]`：最后一行
- `X[1:3]`：第2到第3行
- `X[i, j]`：第i行第j列的元素
- `X[0:2, :]`：前两行的所有列

**赋值操作：**

```python
X[1, 2] = 9       # 修改单个元素
X[0:2, :] = 12    # 修改多个元素
```

### 2.1.5 节省内存

**原地操作 vs 创建新张量：**

- `Y = Y + X`：创建新张量，Y的内存地址改变
- `Y += X`：原地操作，Y的内存地址不变
- `Z[:] = X + Y`：将结果赋值给已存在的张量

```python
# 示例 1: 创建新张量
Y = torch.tensor([1, 2, 3])
before = id(Y)      # 获取 Y 的内存地址，例如 140234567890
Y = Y + X          # 创建新张量，Y 指向新的内存地址
print(id(Y) == before)  # False，内存地址改变了

# 示例 2: 原地操作
X = torch.tensor([1, 2, 3])
before = id(X)      # 获取 X 的内存地址
X += Y             # 原地操作，X 仍在原来的内存地址
print(id(X) == before)  # True，内存地址不变
```

**关于 `id()` 函数的说明：**

`id()` 是 **Python 内置函数**，不是 PyTorch 的方法。它返回对象的**唯一标识符**，在 CPython 实现中，这个标识符就是对象在内存中的**地址**。

**为什么需要查看内存地址？**

在 PyTorch 中，区分**原地操作**和**创建新张量**非常重要，原因如下：

1. **内存效率**：原地操作不会分配新内存，节省内存；创建新张量会分配新内存，可能导致内存不足
2. **梯度追踪**：原地操作可能会破坏计算图，影响反向传播；创建新张量可以保持计算图的完整性
3. **性能优化**：原地操作通常更快，因为不需要内存分配和复制

**实际应用场景：**

```python
# 场景 1: 避免内存泄漏
result = torch.zeros(1000)
# 不好的做法：循环中不断创建新张量
for i in range(1000):
    result = result + x  # 每次都创建新张量，内存占用增加

# 好的做法：使用原地操作
for i in range(1000):
    result += x  # 原地操作，内存占用不变

# 场景 2: 梯度计算时的注意事项
x = torch.tensor([1.0, 2.0], requires_grad=True)
x += 1  # 原地操作，会报错！因为破坏了计算图
# RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.

# 场景 3: 验证共享内存
x = torch.tensor([1, 2, 3])
y = x.view(3, 1)  # view 操作共享内存
print(id(x.data) == id(y.data))  # True，共享相同的内存

z = x.clone()  # clone 创建新副本
print(id(x.data) == id(z.data))  # False，不同的内存
```

**总结：**

- `id()` 是 Python 内置函数，返回对象的内存地址
- 用于验证操作是否创建了新对象还是原地修改
- 在深度学习中，理解内存管理对性能优化至关重要
- 原地操作节省内存但可能影响梯度计算
- 创建新张量更安全但消耗更多内存

### 2.1.6 张量与其他Python对象的转换

**转换方法：**

- `.numpy()`：张量转NumPy数组
- `torch.tensor(array)`：NumPy数组转张量
- `.item()`：单元素张量转Python标量
- `float(tensor)`：转Python浮点数
- `int(tensor)`：转Python整数

```python
A = X.numpy()           # 转NumPy数组
B = torch.tensor(A)     # 转回张量
a = torch.tensor([3.5])
print(a.item())         # 3.5
print(float(a))         # 3.5
print(int(a))           # 3
```

---

## 2.2 数据预处理

数据预处理是机器学习的重要环节，包括读取数据、处理缺失值和格式转换。

### 2.2.1 读取数据集

使用pandas库读取CSV文件：

```python
import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(__file__), "data")
data = pd.read_csv(os.path.join(data_dir, "house_tiny.csv"))
```

### 2.2.2 处理缺失值

**处理方法：**

- `.fillna(value)`：用指定值填充缺失值
- `.mean()`：计算列的平均值
- `.iloc[:, start:end]`：按位置选择列

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())  # 用平均值填充缺失值
```

### 2.2.3 转换为张量格式

将pandas DataFrame转换为PyTorch张量：

```python
import torch

X = torch.tensor(inputs.values)   # 输入特征张量
y = torch.tensor(outputs.values)  # 输出标签张量
```

---

## 2.3 线性代数

线性代数是深度学习的数学基础，PyTorch提供了丰富的线性代数运算。

### 2.3.1 标量

标量是0维张量，表示单个数值：

```python
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y, x * y, x / y, x ** y)
```

### 2.3.2 向量

向量是1维张量：

```python
x = torch.arange(4)  # [0, 1, 2, 3]
print(x[3])          # 第4个元素
print(len(x))        # 向量长度：4
print(x.shape)       # 形状：torch.Size([4])
```

### 2.3.3 矩阵

矩阵是2维张量：

```python
A = torch.arange(20).reshape(5, 4)  # 5x4矩阵
print(A.shape)      # torch.Size([5, 4])
print(A.numel())    # 元素总数：20
print(A.T)          # 转置矩阵
```

**对称矩阵：**矩阵等于其转置，即A = A.T

### 2.3.4 张量

张量可以是任意维度的数组：

```python
X = torch.arange(24).reshape(2, 3, 4)  # 2x3x4的三维张量
```

### 2.3.5 张量算法的基本性质

**逐元素运算：**

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A + B)  # 逐元素相加
print(A * B)  # 逐元素相乘
```

**标量与张量的运算（广播机制）：**

```python
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)  # 标量与张量相加
print(a * X)  # 标量与张量相乘
```

### 2.3.6 降维

**求和与平均：**

```python
x = torch.arange(4, dtype=torch.float32)
print(x.sum())  # 所有元素的和

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A.sum(axis=0))  # 沿维度0（行）求和
print(A.sum(axis=1))  # 沿维度1（列）求和
print(A.sum(axis=[0, 1]))  # 所有元素的和
print(A.mean())  # 所有元素的平均值
```

**保持维度：**

```python
sum_A = A.sum(axis=1, keepdims=True)  # 保持维度
print(sum_A.shape)  # torch.Size([5, 1])
print(A / sum_A)  # 归一化
```

**累积和：**

```python
print(A.cumsum(axis=0))  # 沿维度0的累积和
```

### 2.3.7 点积

点积是向量对应元素乘积的和：

```python
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(torch.dot(x, y))  # 点积
print(torch.sum(x * y))  # 等价计算
```

### 2.3.8 矩阵-向量积

矩阵与向量的乘积：

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.arange(4, dtype=torch.float32)
print(torch.mv(A, x))  # 矩阵-向量积
```

### 2.3.9 矩阵-矩阵乘法

矩阵乘法（不是逐元素乘法）：

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = torch.ones(4, 3)
print(torch.mm(A, B))  # 矩阵乘法
```

### 2.3.10 范数

**常用范数：**

- L2范数：向量元素的平方和的平方根
- L1范数：向量元素的绝对值之和
- Frobenius范数：矩阵元素的平方和的平方根

```python
x = torch.arange(4, dtype=torch.float32)
print(torch.norm(x))  # L2范数
print(torch.abs(x).sum())  # L1范数
print(torch.norm(torch.ones((4, 9))))  # Frobenius范数
```

---

## 2.4 微积分

微积分是理解深度学习优化算法的基础。

### 2.4.1 数值极限

通过差分商近似计算导数：

```python
def numerical_lim(f, x, h):
    return (f(x+h) - f(x)) / h

def f(x):
    return 3*x**2 - 4*x

h = 0.1
for i in range(5):
    print(f"h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}")
    h *= 0.1
```

当h趋近于0时，差分商趋近于导数。

### 2.4.2 可视化函数

**绘图工具函数：**

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline

def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, 
         xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or
                isinstance(X, list) and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

**绘制函数和切线：**

```python
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2*x-3], 'x', 'f(x)', 
     legend=['f(x)', 'Tangent line (x=1)'])
plt.show()
```

---

## 2.5 自动微分

自动微分是深度学习框架的核心功能，用于计算梯度。

### 2.5.1 基本梯度计算

**设置梯度追踪：**

```python
x = torch.arange(4.0)
x.requires_grad_(True)  # 标记需要计算梯度
```

**计算梯度：**

```python
y = 2 * torch.dot(x, x)  # y = 2 * (x·x) = 2 * Σx²
y.backward()  # 反向传播计算梯度
print(x.grad)  # dy/dx = 4x
```

**梯度清零：**

```python
x.grad.zero_()  # 清零梯度
y = x.sum()
y.backward()
print(x.grad)  # dy/dx = 1
```

### 2.5.2 非标量变量的反向传播

对于非标量输出，需要先求和再反向传播：

```python
x.grad.zero_()
y = x * x  # y = [0, 1, 4, 9]
y.sum().backward()  # 对y求和后计算梯度
print(x.grad)  # dy/dx = 2x = [0, 2, 4, 6]
```

### 2.5.3 分离计算

使用`.detach()`分离计算图，阻断梯度传播：

```python
x.grad.zero_()
y = x * x
u = y.detach()  # 分离y，u不参与梯度计算
z = u * x
z.sum().backward()
print(x.grad)  # dz/dx = u = y = x*x

x.grad.zero_()
y.sum().backward()
print(x.grad)  # dy/dx = 2x
```

### 2.5.4 控制流的梯度计算

自动微分可以处理包含控制流的函数：

```python
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad)  # 自动计算梯度
```

---

## 2.6 概率

概率论是理解机器学习模型的基础。

### 2.6.1 多项式分布

使用多项式分布模拟随机实验：

```python
from torch.distributions import multinomial

fair_probs = torch.ones([6]) / 6  # 均匀分布
counts = multinomial.Multinomial(10, fair_probs).sample()
print(counts)  # 10次投掷的结果
```

### 2.6.2 大数定律

随着实验次数增加，频率趋近于概率：

```python
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts /= 1000  # 转换为频率
print(counts)  # 频率接近1/6
```

### 2.6.3 频率估计的可视化

通过多次实验观察频率收敛过程：

```python
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)  # 累积计数
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

plt.rcParams['figure.figsize'] = (6, 6)
for i in range(6):
    plt.plot(estimates[:, i].numpy(), 
             label=("P(die=" + str(i + 1) + ")"))
plt.axhline(y=1/6, color='black', linestyle='--')
plt.legend()
plt.show()
```

随着实验次数增加，估计频率逐渐收敛到理论概率1/6。

---

## 2.7 查阅文档

熟练查阅文档是高效开发的关键。

### 2.7.1 查找模块中的所有函数和类

使用`dir()`函数查看模块内容：

```python
import torch
print(dir(torch.distributions))
```

### 2.7.2 查找特定函数和类的用法

使用`help()`函数查看详细文档：

```python
print(help(torch.ones))
print("torch.ones(4):", torch.ones(4))
```

**其他文档查阅方法：**

- 官方文档：https://pytorch.org/docs/
- Jupyter Notebook中的`?`和`??`操作符
- 在线搜索和社区资源

---

## 总结

本章介绍了深度学习所需的预备知识：

1. **数据操作**：张量的创建、运算、索引、广播和内存管理
2. **数据预处理**：读取数据、处理缺失值、格式转换
3. **线性代数**：标量、向量、矩阵、张量及其运算
4. **微积分**：数值极限、导数、函数可视化
5. **自动微分**：梯度计算、反向传播、计算图管理
6. **概率**：概率分布、大数定律、频率估计
7. **查阅文档**：高效查找和使用API文档

这些基础知识为后续学习深度学习模型奠定了坚实基础。
