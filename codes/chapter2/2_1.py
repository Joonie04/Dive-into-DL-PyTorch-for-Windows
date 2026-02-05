# 2.1 数据操作
import torch  # 导入PyTorch库


# 2.1.1. 张量的操作
x = torch.arange(12)  # 创建一个包含0到11的张量
print("x:", x)  # 打印张量x
X = x.reshape(3, 4)  # 将张量x重塑为3行4列, 即3x4的矩阵
print("X:", X)  # 打印重塑后的张量X
Z = torch.zeros((2, 3, 4))  # 创建一个形状为(2,3,4)的全零张量
print("Z:", Z)  # 打印全零张量Z
Y = torch.ones((3, 4))  # 创建一个形状为(3,4)的全一张量
print("Y:", Y)  # 打印全一张量Y
R = torch.rand((3, 4))  # 创建一个形状为(3,4)的随机张量
print("R:", R)  # 打印随机张量R
X = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])  # 直接创建一个张量
print("X:", X)  # 打印张量X

print("-" * 50)  # 打印分隔线

# 张量的基本属性
print("X的形状:", X.shape)  # 打印张量X的形状
print("X的元素总数:", X.numel())  # 打印张量X的元素总数
print("X的元素类型:", X.dtype)  # 打印张量X的元素类型
print("X的维度数:", X.ndim)  # 打印张量X的维度数
print("X的设备:", X.device)  # 打印张量X所在的设备
print("X是否连续:", X.is_contiguous())  # 检查张量是否在内存中连续存储

print("-" * 50)  # 打印分隔线

# 形状变换
X_squeezed = X.unsqueeze(0)  # 在维度0添加长度为1的轴
# 说明
# X.shape = 3x4
# X.unsqueeze(0)的形状 = 1x3x4
print("X.unsqueeze(0)的形状:", X_squeezed.shape)  # 打印添加维度后的形状
print("X.unsqueeze(0)的内容:", X_squeezed)  # 打印添加维度后的内容

X_squeezed = X.unsqueeze(1)  # 在维度1添加长度为1的轴
# 说明
# X.shape = 3x4
# X.unsqueeze(1)的形状 = 3x1x4
print("X.unsqueeze(1)的形状:", X_squeezed.shape)  # 打印添加维度后的形状
print("X.unsqueeze(1)的内容:", X_squeezed)  # 打印添加维度后的内容

print("-" * 50)  # 打印分隔线

# 展平操作
X_flattened = X.flatten()  # 将张量展平为1D
print("X.flatten()的形状:", X_flattened.shape)  # 打印展平后的形状
print("X.flatten()的内容:", X_flattened)  # 打印展平后的内容

X_flattened_partial = X.flatten(start_dim=1)  # 从维度1开始展平
print("X.flatten(start_dim=1)的形状:", X_flattened_partial.shape)  # 打印部分展平后的形状
print("X.flatten(start_dim=1)的内容:", X_flattened_partial)  # 打印部分展平后的内容

# 3D张量的flatten示例1
X_3d = torch.arange(24).reshape(2, 3, 4)  # 创建2x3x4的3D张量
print("\n3D张量示例:", X_3d)
print("X_3d.shape:", X_3d.shape)  # (2, 3, 4)
print("X_3d.flatten(start_dim=1).shape:", X_3d.flatten(start_dim=1).shape)  # (2, 12)，展平后两维
print("X_3d.flatten(start_dim=1)的内容:", X_3d.flatten(start_dim=1))  # 打印部分展平后的内容
print("X_3d.flatten(start_dim=2).shape:", X_3d.flatten(start_dim=2).shape)  # (2, 3, 4)，无变化
print("X_3d.flatten(start_dim=2)的内容:", X_3d.flatten(start_dim=2))  # 打印部分展平后的内容

# 3D张量的flatten示例2
X_3d = torch.arange(36).reshape(3, 3, 4)  # 创建3x3x4的3D张量
print("\n3D张量示例:", X_3d)
print("X_3d.shape:", X_3d.shape)  # (3, 3, 4)
print("X_3d.flatten(start_dim=1).shape:", X_3d.flatten(start_dim=1).shape)  # (3, 12)，展平后两维
print("X_3d.flatten(start_dim=1)的内容:", X_3d.flatten(start_dim=1))  # 打印部分展平后的内容
print("X_3d.flatten(start_dim=2).shape:", X_3d.flatten(start_dim=2).shape)  # (3, 12)，无变化
print("X_3d.flatten(start_dim=2)的内容:", X_3d.flatten(start_dim=2))  # 打印部分展平后的内容

print("-" * 50)  # 打印分隔线

# 维度操作
X_transposed = X.transpose(0, 1)  # 交换维度0和维度1
print("X.transpose(0, 1)的形状:", X_transposed.shape)  # 打印转置后的形状
print("X的内容:", X)  # 打印转置后的内容
print("X.transpose(0, 1)的内容:", X_transposed)  # 打印转置后的内容
X_permuted = X.permute(1, 0)  # 按照指定顺序重新排列维度
print("X.permute(1, 0)的形状:", X_permuted.shape)  # 打印重排后的形状
print("X.permute(1, 0)的内容:", X_permuted)  # 打印重排后的内容
X_t = X.t()  # 矩阵转置（仅适用于2D张量）
print("X.t()的形状:", X_t.shape)  # 打印转置后的形状
print("X.t()的内容:", X_t)  # 打印转置后的内容

print("-" * 50)  # 打印分隔线

# 复制和扩展
X_cloned = X.clone()  # 创建张量的深拷贝
print("X.clone()是否为副本:", X_cloned is not X)  # 检查是否为副本
print("X_cloned的内容:", X_cloned)  # 打印深拷贝后的内容
print("X的内容:", X)  # 打印转置后的内容

X_repeated = X.repeat(2, 1)  # 沿维度0重复2次，维度1重复1次
print("X.repeat(2, 1)的形状:", X_repeated.shape)  # 打印重复后的形状
print("X.repeat(2, 1)的内容:", X_repeated)  # 打印重复后的内容
print("X的内容:", X)  # 打印转置后的内容

X_expanded = X.unsqueeze(0).expand(3, -1, -1)  # 扩展张量（不复制数据）
print("X.expand(3, -1, -1)的形状:", X_expanded.shape)  # 打印扩展后的形状
print("X.expand(3, -1, -1)的内容:", X_expanded)  # 打印扩展后的内容
print("X的内容:", X)  # 打印转置后的内容

print("-" * 50)  # 打印分隔线

# 类型转换
X_float = X.float()  # 转换为float32类型
print("X.float()的类型:", X_float.dtype)  # 打印转换后的类型
X_int = X.int()  # 转换为int32类型
print("X.int()的类型:", X_int.dtype)  # 打印转换后的类型
X_long = X.long()  # 转换为int64类型
print("X.long()的类型:", X_long.dtype)  # 打印转换后的类型
X_double = X.double()  # 转换为float64类型
print("X.double()的类型:", X_double.dtype)  # 打印转换后的类型
X_bool = X > 0  # 转换为bool类型（元素大于0为True，否则为False）
print("X > 0 的结果:", X_bool)  # 打印转换后的内容

print("-" * 50)  # 打印分隔线

# 其他转换
X_list = X.tolist()  # 将张量转换为Python列表
print("X.tolist()的类型:", type(X_list))  # 打印转换后的类型
X_numpy = X.numpy()  # 将张量转换为NumPy数组
print("X.numpy()的类型:", type(X_numpy))  # 打印转换后的类型

print("-" * 50)  # 打印分隔线

# 2.1.2 张量的运算
x = torch.tensor(([1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]))  # 创建一个3x4的张量x
y = torch.tensor(([2,3,2,2], [2,3,2,2], [2,3,2,2]))  # 创建一个3x4的张量y
print(x)  # 打印张量x
print(y)  # 打印张量y
print("加运算", x+y)  # 张量逐元素相加
print("减运算", x-y)  # 张量逐元素相减
print("乘运算", x*y)  # 张量逐元素相乘
print("除运算", x/y)  # 张量逐元素相除
print("整除运算", x//y)  # 张量逐元素整除（向下取整）
print("取余运算", x%y)  # 张量逐元素取余
print("乘方运算", x**y)  # 张量逐元素乘方
print("exp运算", torch.exp(x))  # 张量逐元素计算指数
print("log运算", torch.log(x))  # 张量逐元素计算自然对数
print("sqrt运算", torch.sqrt(x))  # 张量逐元素计算平方根
print("abs运算", torch.abs(x))  # 张量逐元素计算绝对值
print("sign运算", torch.sign(x))  # 张量逐元素计算符号, 大于0为1, 小于0为-1, 等于0为0
print("ceil运算", torch.ceil(x))  # 张量逐元素向上取整
print("floor运算", torch.floor(x))  # 张量逐元素向下取整
print("round运算", torch.round(x))  # 张量逐元素四舍五入
print("sin运算", torch.sin(x))  # 张量逐元素计算正弦
print("cos运算", torch.cos(x))  # 张量逐元素计算余弦
print("大于比较", x>y)  # 逐元素比较x是否大于y
print("小于比较", x<y)  # 逐元素比较x是否小于y
print("等于比较", x==y)  # 逐元素比较x是否等于y
print("不等于比较", x!=y)  # 逐元素比较x是否不等于y

# 聚合运算（需要浮点类型）
x_float = x.float()  # 转换为浮点类型以便计算统计量
print("最大值", x_float.max())  # 计算x的最大值
print("最小值", x_float.min())  # 计算x的最小值
print("平均值", x_float.mean())  # 计算x的平均值
print("标准差", x_float.std())  # 计算x的标准差
print("方差", x_float.var())  # 计算x的方差
print("最大值索引", x_float.argmax())  # 返回最大值的索引
print("最小值索引", x_float.argmin())  # 返回最小值的索引

print("-" * 50)  # 打印分隔线

X = torch.arange(12, dtype=torch.float32).reshape((3,4))  # 创建一个3x4的float32类型张量
Y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])  # 创建一个3x4的张量
Z1 = torch.cat((X, Y), dim=0)  # 沿维度0（行）拼接张量X和Y
Z2 = torch.cat((X, Y), dim=1)  # 沿维度1（列）拼接张量X和Y
Z3 = torch.stack((X, Y), dim=0)  # 沿新维度0堆叠张量X和Y
Z4 = torch.stack((X, Y), dim=1)  # 沿新维度1堆叠张量X和Y
Z5 = torch.stack((X, Y), dim=2)  # 沿新维度2堆叠张量X和Y
print("X:", X)  # 打印张量X
print("X.shape:", X.shape)  # 打印张量X的形状
print("Y:", Y)  # 打印张量Y
print("Y.shape:", Y.shape)  # 打印张量Y的形状
print("Z1 (沿维度0拼接):", Z1)  # 打印沿维度0拼接的结果
print("Z1.shape (沿维度0拼接):", Z1.shape)  # 打印沿维度0拼接的结果
print("Z2 (沿维度1拼接):", Z2)  # 打印沿维度1拼接的结果
print("Z2.shape (沿维度1拼接):", Z2.shape)  # 打印沿维度1拼接的结果
print("Z3 (沿新维度0堆叠):", Z3)  # 打印沿新维度0堆叠的结果
print("Z3.shape (沿新维度0堆叠):", Z3.shape)  # 打印沿新维度0堆叠的结果
print("Z4 (沿新维度1堆叠):", Z4)  # 打印沿新维度1堆叠的结果
print("Z4.shape (沿新维度1堆叠):", Z4.shape)  # 打印沿新维度1堆叠的结果
print("Z5 (沿新维度2堆叠):", Z5)  # 打印沿新维度2堆叠的结果
print("Z5.shape (沿新维度2堆叠):", Z5.shape)  # 打印沿新维度2堆叠的结果

print("X==Y:", X==Y)  # 逐元素比较X和Y是否相等
print("X.sum():", X.sum())  # 计算X中所有元素的和
print("X.prod():", X.prod())  # 计算X中所有元素的乘积

print("-" * 50)  # 打印分隔线

# 2.1.3 张量的广播机制
x = torch.arange(3).reshape((3, 1))  # 创建一个3x1的张量
y = torch.arange(2).reshape((1, 2))  # 创建一个1x2的张量
print("x:", x)  # 打印张量x
print("y:", y)  # 打印张量y
print("广播机制:", x + y)  # 广播后相加，得到3x2的张量

print("-" * 50)  # 打印分隔线

# 2.1.4 张量的索引和切片
print("X:", X)  # 打印X
print("X[-1]:", X[-1])  # 打印X的最后一行
print("X[1:3]:", X[1:3])  # 打印X的第2到第3行

X[1, 2] = 9  # 将X的第2行第3列的元素设置为9
print("X:", X)  # 打印修改后的X

X[0:2, :] = 12  # 将X的前两行的所有元素设置为12
print("X:", X)  # 打印修改后的X

print("-" * 50)  # 打印分隔线

# 2.1.5 节省内存
before = id(Y)  # 获取Y的内存地址
Y = Y + X  # 执行加法运算，这会创建新的张量
print("id(Y) == before", id(Y) == before)  # 比较Y的内存地址是否改变

Z = torch.zeros_like(Y)  # 创建一个形状和类型与Y相同的全零张量
print("Z:", Z)  # 打印Z
Z[:] = X + Y  # 将X+Y的结果赋值给Z的所有元素, 不创建新的张量. 原地赋值; 等价于Z = X + Y, 但是Z = X + Y会创建新的张量, 而Z[:] = X + Y不会创建新的张量.
print("Z:", Z)  # 打印Z

before = id(X)  # 获取X的内存地址
X += Y  # 使用原地操作，不会创建新的张量
print("id(X) == before:", id(X) == before)  # 比较X的内存地址是否改变

print("-" * 50)  # 打印分隔线

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

print("-" * 50)  # 打印分隔线