# 2.3 线性代数

# 2.3.1 标量
import torch  # 导入PyTorch库

x = torch.tensor(3.0)  # 创建一个标量张量x
y = torch.tensor(2.0)  # 创建一个标量张量y

print("x:", x)  # 打印x
print("y:", y)  # 打印y
print("x+y:", x+y)  # 打印x和y的和
print("x*y:", x*y)  # 打印x和y的积
print("x/y:", x/y)  # 打印x除以y
print("x**y:", x**y)  # 打印x的y次方


# 2.3.2 向量
x = torch.arange(4)  # 创建一个包含0到3的向量
print("x:", x)  # 打印向量x
print("x[3]:", x[3])  # 打印向量的第4个元素（索引为3）
print("len(x):", len(x))  # 打印向量的长度
print("x.shape:", x.shape)  # 打印向量的形状


# 2.3.3 矩阵
A = torch.arange(20).reshape(5, 4)  # 创建一个5行4列的矩阵
print("A:", A)  # 打印矩阵A
print("A.shape:", A.shape)  # 打印矩阵A的形状
print("A.numel():", A.numel())  # 打印矩阵A的元素总数
print("A.T:", A.T)  # 打印矩阵A的转置
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])  # 创建一个3x3的对称矩阵
print("B:", B)  # 打印矩阵B
print("B==B.T:", B==B.T)  # 检查矩阵B是否等于其转置（是否为对称矩阵）


# 2.3.4 张量
X = torch.arange(24).reshape(2, 3, 4)  # 创建一个2x3x4的三维张量
print("X:", X)  # 打印张量X


# 2.3.5 张量算法的基本性质
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)  # 创建一个5x4的float32类型矩阵
B = A.clone()  # 克隆矩阵A得到矩阵B
print("A:", A)  # 打印矩阵A
print("B:", B)  # 打印矩阵B
print("A+B:", A+B)  # 打印矩阵A和B的和
print("A*B:", A*B)  # 打印矩阵A和B的逐元素乘积

a = 2  # 定义一个标量a
X = torch.arange(24).reshape(2, 3, 4)  # 创建一个2x3x4的张量
print("a + X:", a + X)  # 打印标量a与张量X的和（广播机制）
print("(a*X).shape:", (a*X).shape)  # 打印标量a与张量X积的形状


# 2.3.6 降维
x = torch.arange(4, dtype=torch.float32)  # 创建一个float32类型的向量
print("x :", x)  # 打印向量x
print("x.sum():", x.sum())  # 打印向量x所有元素的和
print("A.shape:", A.shape)  # 打印矩阵A的形状
print("A.sum():", A.sum())  # 打印矩阵A所有元素的和

A_sum_axis0 = A.sum(axis=0)  # 沿维度0（行）求和
print("A_sum_axis0:", A_sum_axis0)  # 打印沿维度0求和的结果
print("A_sum_axis0.shape:", A_sum_axis0.shape)  # 打印结果的形状

A_sum_axis1 = A.sum(axis=1)  # 沿维度1（列）求和
print("A_sum_axis1:", A_sum_axis1)  # 打印沿维度1求和的结果
print("A_sum_axis1.shape:", A_sum_axis1.shape)  # 打印结果的形状

print("A.sum(axis=[0, 1]):", A.sum(axis=[0, 1]))  # 沿维度0和1求和（所有元素的和）
print("A.mean():", A.mean())  # 打印矩阵A所有元素的平均值
print("A.sum()/A.numel():", A.sum()/A.numel())  # 打印总和除以元素总数（平均值）
print("A.mean(axis=0):", A.mean(axis=0))  # 打印沿维度0的平均值
print("A.sum(axis=0)/A.shape[0]:", A.sum(axis=0)/A.shape[0])  # 打印沿维度0求和后除以行数（平均值）

sum_A = A.sum(axis=1, keepdims=True)  # 沿维度1求和并保持维度
print("sum_A:", sum_A)  # 打印求和结果
print("sum_A.shape:", sum_A.shape)  # 打印结果的形状
print("A/sum_A:", A/sum_A)  # 打印A除以sum_A（归一化）
print("A.cumsum(axis=0):", A.cumsum(axis=0))  # 打印沿维度0的累积和


# 2.3.7 点积
x = torch.arange(4, dtype=torch.float32)  # 创建一个float32类型的向量
y = torch.ones(4, dtype=torch.float32)  # 创建一个全1的float32类型向量
print("x:", x)  # 打印向量x
print("y:", y)  # 打印向量y
print("x.dot(y):", x.dot(y))  # 打印x和y的点积
print("torch.dot(x, y):", torch.dot(x, y))  # 使用torch.dot计算点积
print("torch.sum(x*y):", torch.sum(x*y))  # 通过逐元素乘积求和计算点积


# 2.3.8 矩阵-向量积
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)  # 创建一个5x4的矩阵
x = torch.arange(4, dtype=torch.float32)  # 创建一个长度为4的向量
print("A:", A)  # 打印矩阵A
print("A.shape:", A.shape)  # 打印矩阵A的形状
print("x:", x)  # 打印向量x
print("x.shape:", x.shape)  # 打印向量x的形状
print("torch.mv(A, x):", torch.mv(A, x))  # 打印矩阵A与向量x的乘积


# 2.3.9 矩阵-矩阵乘法
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)  # 创建一个5x4的矩阵
B = torch.ones(4, 3)  # 创建一个4x3的全1矩阵
print("A:", A)  # 打印矩阵A
print("A.shape:", A.shape)  # 打印矩阵A的形状
print("B:", B)  # 打印矩阵B
print("B.shape:", B.shape)  # 打印矩阵B的形状
print("torch.mm(A, B):", torch.mm(A, B))  # 打印矩阵A与矩阵B的乘积


# 2.3.10 范数
x = torch.arange(4, dtype=torch.float32)  # 创建一个float32类型的向量
print("x:", x)  # 打印向量x
print("torch.norm(x):", torch.norm(x))  # 打印向量x的L2范数
print("torch.abs(x).sum():", torch.abs(x).sum())  # 打印向量x的L1范数
print("torch.norm(torch.ones((4, 9))):", torch.norm(torch.ones((4, 9))))  # 打印4x9全1矩阵的Frobenius范数
print("torch.sqrt(torch.sum(torch.square(x))):", torch.sqrt(torch.sum(torch.square(x))))  # 手动计算L2范数