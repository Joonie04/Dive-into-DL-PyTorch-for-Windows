# 2.1 数据操作
import torch  # 导入PyTorch库

# 2.1.1. 张量的操作
x = torch.arange(12)  # 创建一个包含0到11的张量
print("x:", x)  # 打印张量x
print("x的形状:", x.shape)  # 打印张量x的形状
print("x的元素总数:", x.numel())  # 打印张量x的元素总数

X = x.reshape(3, 4)  # 将张量x重塑为3行4列
print("X:", X)  # 打印重塑后的张量X
Z = torch.zeros((2, 3, 4))  # 创建一个形状为(2,3,4)的全零张量
print("Z:", Z)  # 打印全零张量Z
Y = torch.ones((3, 4))  # 创建一个形状为(3,4)的全一张量
print("Y:", Y)  # 打印全一张量Y
R = torch.rand((3, 4))  # 创建一个形状为(3,4)的随机张量
print("R:", R)  # 打印随机张量R
X = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])  # 直接创建一个张量
print("X:", X)  # 打印张量X

# 2.1.2 张量的运算
x = torch.tensor(([1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]))  # 创建一个3x4的张量x
y = torch.tensor(([2,3,2,2], [2,3,2,2], [2,3,2,2]))  # 创建一个3x4的张量y
print(x)  # 打印张量x
print(y)  # 打印张量y
print("加运算", x+y)  # 张量逐元素相加
print("减运算", x-y)  # 张量逐元素相减
print("乘运算", x*y)  # 张量逐元素相乘
print("除运算", x/y)  # 张量逐元素相除
print("乘方运算", x**y)  # 张量逐元素乘方
print("exp运算", torch.exp(x))  # 张量逐元素计算指数

X = torch.arange(12, dtype=torch.float32).reshape((3,4))  # 创建一个3x4的float32类型张量
Y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])  # 创建一个3x4的张量
Z1 = torch.cat((X, Y), dim=0)  # 沿维度0（行）拼接张量X和Y
Z2 = torch.cat((X, Y), dim=1)  # 沿维度1（列）拼接张量X和Y
print("X:", X)  # 打印张量X
print("Y:", Y)  # 打印张量Y
print("Z1:", Z1)  # 打印沿维度0拼接的结果
print("Z2:", Z2)  # 打印沿维度1拼接的结果

print("X==Y:", X==Y)  # 逐元素比较X和Y是否相等
print("X.sum():", X.sum())  # 计算X中所有元素的和

# 2.1.3 张量的广播机制
x = torch.arange(3).reshape((3, 1))  # 创建一个3x1的张量
y = torch.arange(2).reshape((1, 2))  # 创建一个1x2的张量
print("x:", x)  # 打印张量x
print("y:", y)  # 打印张量y
print("广播机制:", x + y)  # 广播后相加，得到3x2的张量

# 2.1.4 张量的索引和切片
print("X[-1]:", X[-1])  # 打印X的最后一行
print("X[1:3]:", X[1:3])  # 打印X的第2到第3行

X[1, 2] = 9  # 将X的第2行第3列的元素设置为9
print("X:", X)  # 打印修改后的X

X[0:2, :] = 12  # 将X的前两行的所有元素设置为12
print("X:", X)  # 打印修改后的X

# 2.1.5 节省内存
before = id(Y)  # 获取Y的内存地址
Y = Y + X  # 执行加法运算，这会创建新的张量
print(id(Y) == before)  # 比较Y的内存地址是否改变

Z = torch.zeros_like(Y)  # 创建一个形状和类型与Y相同的全零张量
print("Z:", Z)  # 打印Z
Z[:] = X + Y  # 将X+Y的结果赋值给Z的所有元素
print("Z:", Z)  # 打印Z

before = id(X)  # 获取X的内存地址
X += Y  # 使用原地操作，不会创建新的张量
print("id(X) == before:", id(X) == before)  # 比较X的内存地址是否改变

# 2.1.6 转换为其他Python对象
A = X.numpy()  # 将张量X转换为NumPy数组
B = torch.tensor(A)  # 将NumPy数组A转换为张量
print("type(A):", type(A))  # 打印A的类型
print("type(B):", type(B))  # 打印B的类型

a = torch.tensor([3.5])  # 创建一个包含单个元素的张量
print("a:", a)  # 打印张量a
print("a.item():", a.item())  # 将张量转换为Python标量
print("float(a):", float(a))  # 将张量转换为Python浮点数
print("int(a):", int(a))  # 将张量转换为Python整数
