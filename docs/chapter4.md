# 第4章 多层感知机

本章将深入介绍深度学习中最基础的神经网络结构——多层感知机（MLP），并讨论模型训练中常见的问题及其解决方案。

---

## 4.1 激活函数

激活函数是神经网络中的核心组件，它为网络引入非线性特性，使神经网络能够学习复杂的模式和映射关系。没有激活函数，多层神经网络将等价于单层线性模型。

### 4.1.1 激活函数的作用

- **引入非线性**：使网络能够逼近任意复杂函数
- **特征转换**：将输入信号转换为更适合后续处理的形式
- **梯度传播**：影响反向传播中梯度的流动

---

### 4.1.2 常用激活函数详解

#### 1. ReLU (Rectified Linear Unit)

**公式**：`ReLU(x) = max(0, x)`

**特点**：
- 当输入为正时，输出等于输入；当输入为负时，输出为0
- 计算极其简单，只需比较操作

**优点**：
- 计算效率高，收敛速度快
- 有效缓解梯度消失问题（正区间梯度恒为1）
- 具有稀疏激活特性，增强模型泛化能力

**缺点**：
- **死亡ReLU问题**：负区间梯度为0，神经元可能永久失活
- 输出非零中心，可能导致梯度更新偏向

**使用场景**：
- 深度卷积神经网络（如 ResNet、VGG）
- 大多数隐藏层的默认选择
- 需要快速训练的场景

---

#### 2. Sigmoid

**公式**：`σ(x) = 1 / (1 + e^(-x))`

**特点**：
- 输出范围 (0, 1)，可解释为概率
- 函数平滑，处处可导

**优点**：
- 输出有界，适合概率输出
- 在二分类问题中作为输出层激活函数

**缺点**：
- **梯度消失问题**：两端饱和区梯度接近0
- 输出非零中心，影响梯度下降效率
- 计算指数函数开销较大

**使用场景**：
- 二分类问题的输出层
- 注意力机制中的门控单元
- 需要将输出限制在 (0,1) 范围的场景

---

#### 3. Tanh (双曲正切)

**公式**：`tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**特点**：
- 输出范围 (-1, 1)，零中心
- 是 Sigmoid 函数的缩放版本

**优点**：
- 输出零中心，梯度更新更均衡
- 梯度比 Sigmoid 更陡峭，学习更快

**缺点**：
- 仍存在梯度消失问题
- 计算开销较大

**使用场景**：
- RNN/LSTM 中的隐藏状态
- 需要零中心输出的隐藏层
- 数据预处理中的归一化

---

#### 4. LeakyReLU

**公式**：`LeakyReLU(x) = x if x > 0 else αx`（通常 α = 0.01）

**特点**：
- ReLU 的改进版本，负区间有微小斜率
- 避免神经元完全失活

**优点**：
- 解决死亡 ReLU 问题
- 保留 ReLU 的大部分优点
- 负区间信息得以保留

**缺点**：
- 需要调参选择合适的 α 值
- 负区间梯度仍然很小

**使用场景**：
- 深层网络中 ReLU 效果不佳时
- GAN（生成对抗网络）中常用
- 需要保留负区间信息的任务

---

#### 5. ELU (Exponential Linear Unit)

**公式**：`ELU(x) = x if x > 0 else α(e^x - 1)`

**特点**：
- 负区间使用指数函数，输出平滑过渡
- 输出均值接近零

**优点**：
- 输出接近零中心，加速学习
- 缓解梯度消失问题
- 对噪声具有鲁棒性

**缺点**：
- 计算指数函数开销大
- 存在超参数 α 需要调节

**使用场景**：
- 对训练稳定性要求高的场景
- 深层全连接网络
- 需要平滑激活的场景

---

#### 6. GELU (Gaussian Error Linear Unit)

**公式**：`GELU(x) = x · Φ(x)`，其中 Φ 是标准正态分布的累积分布函数

**特点**：
- 结合了 ReLU 的非线性和概率性
- 在 0 附近平滑过渡，非单调函数

**优点**：
- 在 Transformer 架构中表现优异
- 平滑过渡，梯度稳定
- 当前 NLP 领域的主流选择

**缺点**：
- 计算复杂度较高（涉及高斯误差函数）
- 在某些简单任务上可能过度复杂

**使用场景**：
- **Transformer 模型**（BERT、GPT 等）
- 自然语言处理任务
- 预训练大模型

---

#### 7. SiLU / Swish

**公式**：`SiLU(x) = x · sigmoid(x)`

**特点**：
- 自门控激活函数
- 非单调、平滑、有下界无上界

**优点**：
- 在深层网络中表现优于 ReLU
- 负区间有非零输出，避免信息丢失
- 平滑可导，梯度稳定

**缺点**：
- 计算 sigmoid 开销较大
- 某些情况下不如 ReLU 简单高效

**使用场景**：
- **EfficientNet** 等现代 CNN 架构
- 移动端轻量级网络
- 需要平滑激活的深度网络

---

#### 8. Softplus

**公式**：`Softplus(x) = log(1 + e^x)`

**特点**：
- ReLU 的平滑近似
- 处处可导，无尖点

**优点**：
- 平滑可导，适合需要二阶导数的场景
- 输出恒为正，适合需要正输出的场景
- 无死亡神经元问题

**缺点**：
- 计算开销大
- 输出非零中心
- 正区间梯度恒为1，但饱和慢

**使用场景**：
- 需要平滑激活的数学建模
- 概率模型中确保输出为正
- 强化学习中的价值函数

---

#### 9. Mish

**公式**：`Mish(x) = x · tanh(softplus(x))`

**特点**：
- 自正则的非单调激活函数
- 结合了 Softplus 和 Tanh 的特性

**优点**：
- 在多种任务上超越 ReLU 和 Swish
- 平滑过渡，梯度流动稳定
- 负区间有非零梯度

**缺点**：
- 计算复杂度最高
- 在某些轻量级模型中可能不值得

**使用场景**：
- YOLOv4 等目标检测模型
- 追求最佳性能的场景
- 计算资源充足时

---

#### 10. Hardswish

**公式**：`Hardswish(x) = x · ReLU6(x+3) / 6`

**特点**：
- Swish 的分段线性近似
- 专为移动端优化设计

**优点**：
- 计算效率极高（无指数运算）
- 适合移动端部署
- 保留 Swish 的主要优点

**缺点**：
- 近似引入的误差可能影响精度
- 在非移动场景优势不明显

**使用场景**：
- **MobileNetV3** 等移动端模型
- 边缘设备部署
- 对推理速度要求高的场景

---

### 4.1.3 激活函数选择指南

| 场景 | 推荐激活函数 | 原因 |
|------|-------------|------|
| 隐藏层（通用） | ReLU / LeakyReLU | 计算高效，梯度稳定 |
| 二分类输出层 | Sigmoid | 输出可解释为概率 |
| 多分类输出层 | Softmax | 输出和为1的概率分布 |
| RNN/LSTM | Tanh | 零中心，适合序列建模 |
| Transformer | GELU | 当前最佳实践 |
| 轻量级网络 | SiLU / Hardswish | 平衡性能与效率 |
| GAN | LeakyReLU | 避免梯度消失 |
| 移动端部署 | Hardswish / ReLU | 计算开销低 |

### 4.1.4 实践建议

1. **从 ReLU 开始**：大多数情况下 ReLU 是安全的选择
2. **关注死亡神经元**：如果训练停滞，尝试 LeakyReLU 或 ELU
3. **输出层选择**：根据任务类型选择（Sigmoid/Softmax/线性）
4. **避免 Sigmoid/Tanh 在深层网络隐藏层**：容易梯度消失
5. **现代架构遵循原论文**：Transformer 用 GELU，MobileNet 用 Hardswish
6. **实验验证**：不同任务最优激活函数可能不同

---

## 4.2 多层感知机

### 4.2.1 多层感知机的结构

多层感知机（MLP）是最基础的深度神经网络，由输入层、隐藏层和输出层组成。

**核心公式**：
```
H = σ(XW₁ + b₁)    # 隐藏层计算
O = HW₂ + b₂        # 输出层计算
```

其中 σ 是激活函数，引入非线性。

### 4.2.2 为什么需要隐藏层？

- **线性模型的局限**：单层线性模型只能表示线性关系
- **隐藏层的作用**：通过激活函数引入非线性，使网络能够逼近任意复杂函数
- **通用近似定理**：具有足够多神经元的单隐藏层MLP可以逼近任何连续函数

### 4.2.3 从零开始实现

**关键步骤**：

1. **初始化参数**：
```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs))
```

2. **定义模型**：
```python
def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(X @ W1 + b1)    # 隐藏层 + ReLU激活
    return H @ W2 + b2        # 输出层
```

3. **训练过程**：
   - 使用交叉熵损失函数
   - 使用小批量随机梯度下降优化
   - 迭代多个epoch更新参数

### 4.2.4 简洁实现（使用PyTorch）

```python
net = nn.Sequential(
    nn.Flatten(),           # 展平输入
    nn.Linear(784, 256),    # 第一个全连接层
    nn.ReLU(),              # 激活函数
    nn.Linear(256, 10)      # 输出层
)
```

**关键组件**：
- `nn.Flatten()`：将多维输入展平为一维
- `nn.Linear(in, out)`：全连接层
- `nn.ReLU()`：ReLU激活函数

---

## 4.3 模型选择、欠拟合和过拟合

### 4.3.1 核心概念

| 概念 | 定义 | 表现 |
|------|------|------|
| **欠拟合** | 模型过于简单，无法捕捉数据规律 | 训练误差和测试误差都很高 |
| **过拟合** | 模型过于复杂，学习了噪声 | 训练误差低，测试误差高 |
| **泛化能力** | 模型在未见数据上的表现 | 测试误差接近训练误差 |

### 4.3.2 影响泛化能力的因素

1. **模型复杂度**
   - 参数数量越多，模型越容易过拟合
   - 隐藏层数量和神经元数量

2. **训练数据量**
   - 数据越多，越难过拟合
   - 小数据集需要更简单的模型

3. **特征数量**
   - 特征过多容易导致维度灾难
   - 需要特征选择或降维

### 4.3.3 多项式拟合实验

通过多项式阶数控制模型复杂度：

```python
# 三阶多项式（正常拟合）
train(poly_features[:n_train, :4], ...)

# 线性函数（欠拟合）
train(poly_features[:n_train, :2], ...)

# 高阶多项式（过拟合）
train(poly_features[:n_train, :], ...)
```

**实验结论**：
- 三阶多项式能够很好地拟合数据
- 线性模型（一阶）欠拟合，训练和测试误差都很高
- 高阶多项式过拟合，训练误差低但测试误差高

### 4.3.4 解决策略

| 问题 | 解决方案 |
|------|----------|
| 欠拟合 | 增加模型复杂度、添加特征、减少正则化 |
| 过拟合 | 增加数据、正则化、Dropout、早停 |

---

## 4.4 权重衰减（L2正则化）

### 4.4.1 正则化的原理

**核心思想**：通过限制参数值的大小来降低模型复杂度

**损失函数修改**：
```
L_new = L(w, b) + λ/2 ||w||²
```

其中：
- `L(w, b)` 是原始损失
- `λ` 是正则化超参数
- `||w||²` 是权重向量的L2范数

### 4.4.2 参数更新规则

**原始梯度下降**：
```
w = w - lr * ∂L/∂w
```

**带L2正则化的梯度下降**：
```
w = w - lr * (∂L/∂w + λw)
    = w(1 - lr*λ) - lr * ∂L/∂w
```

**关键洞察**：正则化使权重在每次更新时都会缩小一点，防止权重过大。

### 4.4.3 从零开始实现

```python
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 训练时
l = loss(net(X), y) + lambd * l2_penalty(w)
```

### 4.4.4 简洁实现（PyTorch）

```python
trainer = torch.optim.SGD([
    {"params": net[1].weight, "weight_decay": wd},  # 对权重应用L2正则化
    {"params": net[1].bias}                         # 偏置不正则化
], lr=lr)
```

**注意**：`weight_decay` 参数就是 L2 正则化的 λ 值。

### 4.4.5 实验观察

| 正则化强度 | 权重L2范数 | 过拟合程度 |
|-----------|-----------|-----------|
| λ = 0 | 较大 | 严重过拟合 |
| λ = 3 | 较小 | 过拟合减轻 |

---

## 4.5 Dropout（暂退法）

### 4.5.1 Dropout的原理

**核心思想**：在训练过程中随机"丢弃"一部分神经元，防止神经元过度依赖特定特征。

**数学表达**：
```
h' = h * mask / (1-p)
```

其中：
- `mask` 是随机掩码（以概率 p 为 0，概率 1-p 为 1）
- `p` 是丢弃概率
- 除以 `(1-p)` 是为了保持期望值不变

### 4.5.2 为什么Dropout有效？

1. **集成学习效果**：相当于训练了多个子网络的集成
2. **打破神经元依赖**：防止神经元之间形成复杂的协同适应
3. **增加鲁棒性**：模型不能依赖任何单一特征

### 4.5.3 训练与推理的区别

| 阶段 | Dropout行为 |
|------|------------|
| 训练 | 随机丢弃神经元，输出需要缩放 |
| 推理 | 不丢弃，但输出乘以 (1-p) 或训练时已缩放 |

### 4.5.4 从零开始实现

```python
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

### 4.5.5 简洁实现（PyTorch）

```python
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),      # Dropout层
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.5),      # Dropout层
    nn.Linear(256, 10)
)
```

### 4.5.6 Dropout率的选择

| 层类型 | 推荐Dropout率 |
|--------|--------------|
| 输入层 | 0.1 - 0.2 |
| 隐藏层 | 0.3 - 0.5 |
| 输出层 | 通常不用 |

---

## 4.6 数值稳定性和模型初始化

### 4.6.1 梯度消失

**定义**：在反向传播过程中，梯度逐层变小，导致浅层参数几乎不更新。

**原因**：
- Sigmoid/Tanh 在饱和区梯度接近0
- 多个小梯度相乘导致梯度指数级衰减

**Sigmoid的梯度问题**：
```python
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))
# 当 |x| 较大时，梯度接近0
```

**解决方案**：
- 使用 ReLU 等非饱和激活函数
- 残差连接（ResNet）
- 批量归一化（BatchNorm）
- 合理的权重初始化

### 4.6.2 梯度爆炸

**定义**：梯度在反向传播过程中逐层变大，导致参数更新过大，模型不稳定。

**原因**：
- 大梯度相乘导致梯度指数级增长
- 权重初始化不当

**演示**：
```python
M = torch.normal(0, 1, size=(4, 4))
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))
# M 的元素会变得非常大
```

**解决方案**：
- 梯度裁剪（Gradient Clipping）
- 合理的权重初始化
- 批量归一化

### 4.6.3 参数初始化策略

| 初始化方法 | 公式 | 适用场景 |
|-----------|------|---------|
| **Xavier初始化** | w ~ U(-√(6/(n_in+n_out)), √(6/(n_in+n_out))) | Tanh/Sigmoid |
| **He初始化** | w ~ N(0, √(2/n_in)) | ReLU及其变体 |
| **随机初始化** | w ~ N(0, 0.01) | 简单网络 |

**PyTorch默认初始化**：
```python
nn.init.xavier_uniform_(m.weight)  # Xavier均匀初始化
nn.init.kaiming_normal_(m.weight)  # He正态初始化
```

### 4.6.4 打破对称性

**为什么不能全零初始化**？
- 所有神经元学习相同的特征
- 反向传播时梯度相同，参数更新相同
- 网络等价于单个神经元

**解决方案**：随机初始化打破对称性

---

## 4.7 实战：Kaggle房价预测

### 4.7.1 数据预处理

**步骤**：

1. **数值特征标准化**：
```python
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / x.std())
```

2. **缺失值处理**：
```python
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

3. **类别特征独热编码**：
```python
all_features = pd.get_dummies(all_features, dummy_na=True)
```

### 4.7.2 模型与损失函数

**模型**：简单的线性回归
```python
net = nn.Sequential(nn.Linear(in_features, 1))
```

**损失函数**：均方误差（MSE）
```python
loss = nn.MSELoss()
```

**评估指标**：对数均方根误差（Log RMSE）
```python
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()
```

### 4.7.3 K折交叉验证

**目的**：更可靠地评估模型性能，选择超参数

**实现**：
```python
def get_k_fold_data(k, i, X, y):
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

### 4.7.4 超参数调优

**关键超参数**：
- 学习率（lr）
- 权重衰减（weight_decay）
- 训练轮数（num_epochs）
- 批量大小（batch_size）

**调优策略**：
1. 使用K折交叉验证评估
2. 观察训练曲线判断过拟合/欠拟合
3. 根据验证损失调整超参数

### 4.7.5 提交结果

```python
preds = net(test_features).detach().numpy()
submission = pd.concat([test_data['Id'], pd.Series(preds.reshape(-1))], axis=1)
submission.to_csv('submission.csv', index=False)
```

---

## 4.8 本章小结

### 核心知识点总结

| 主题 | 关键概念 |
|------|---------|
| 激活函数 | ReLU、Sigmoid、Tanh、GELU等，引入非线性 |
| 多层感知机 | 隐藏层+激活函数，通用近似能力 |
| 模型选择 | 欠拟合vs过拟合，泛化能力 |
| 权重衰减 | L2正则化，防止过拟合 |
| Dropout | 随机丢弃，集成学习效果 |
| 数值稳定性 | 梯度消失/爆炸，合理初始化 |

### 实践建议

1. **模型设计**：从简单模型开始，逐步增加复杂度
2. **正则化**：当出现过拟合时，先尝试权重衰减和Dropout
3. **激活函数**：隐藏层首选ReLU，注意死亡神经元问题
4. **初始化**：使用PyTorch默认初始化通常足够
5. **调参策略**：使用交叉验证，观察学习曲线

### 常见问题与解决方案

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 欠拟合 | 训练/测试误差都高 | 增加模型复杂度 |
| 过拟合 | 训练误差低，测试误差高 | 正则化、Dropout、增加数据 |
| 梯度消失 | 浅层参数不更新 | 换ReLU、残差连接、BatchNorm |
| 梯度爆炸 | Loss变成NaN | 梯度裁剪、调整学习率 |
| 训练缓慢 | 收敛速度慢 | 调整学习率、换优化器 |
