# 2.5 自动微分
import torch  # 导入PyTorch库

# 2.5.1 一个简单的例子
x = torch.arange(4.0)  # 创建一个包含0到3的浮点型张量
print("x:", x)  # 打印张量x

x.requires_grad_(True)  # 设置x需要计算梯度
print("x.requires_grad:", x.requires_grad)  # 打印x是否需要计算梯度

y = 2 * torch.dot(x, x)  # 计算2倍的x与x的点积
print("y:", y)  # 打印y

y.backward()  # 计算y对x的梯度
print("x.grad:", x.grad)  # 打印x的梯度，dy/dx = 4x

x.grad == 4 * x  # 检查梯度是否等于4x
print("x.grad == 4 * x:", x.grad == 4 * x)  # 打印比较结果

x.grad.zero_()  # 将x的梯度清零
print("x.grad after zeroing:", x.grad)  # 打印清零后的梯度

y = x.sum()  # 计算x所有元素的和
print("y:", y)  # 打印y
y.backward()  # 计算y对x的梯度
print("x.grad:", x.grad)  # 打印x的梯度，dy/dx = 1

print("-"*20)  # 打印分隔线

# 2.5.2 非标量变量的反向传播
x.grad.zero_()  # 将x的梯度清零
print("x.grad after zeroing:", x.grad)  # 打印清零后的梯度

y = x * x  # 计算x的逐元素平方
print("y:", y)  # 打印y

y.sum().backward()  # 对y求和后计算梯度
print("x.grad after y.sum().backward():", x.grad)  # 打印x的梯度，dy/dx = 2x

print("-"*20)  # 打印分隔线

# 2.5.3 分离计算
x.grad.zero_()  # 将x的梯度清零
print("x.grad after zeroing:", x.grad)  # 打印清零后的梯度
y = x * x  # 计算x的逐元素平方
print("x:", x)  # 打印x
print("y:", y)  # 打印y

u = y.detach()  # 分离y的计算图，得到u
print("u (detached y):", u)  # 打印分离后的u

z = u * x  # 计算z = u * x
print("z:", z)  # 打印z

z.sum().backward()  # 对z求和后计算梯度
print("x.grad after z.sum().backward():", x.grad)  # 打印x的梯度，dz/dx = u

x.grad.zero_()  # 将x的梯度清零
y.sum().backward()  # 对y求和后计算梯度
print("x.grad == 2 * x after y.sum().backward():", x.grad) # 打印比较结果，dy/dx = 2x

print("-"*20)  # 打印分隔线

# 2.5.4 控制流的梯度计算
def f(a):  # 定义函数f
    # 将输入张量a的每个元素乘以2，得到新的张量b
    b = a * 2  # 计算b = a * 2
    
    # 循环条件：当b的L2范数（欧几里得范数）小于1000时继续循环
    # b.norm()计算的是向量b的长度（平方和的平方根）
    while b.norm() < 1000:  # 当b的范数小于1000时循环
        # 在循环中，每次将b的所有元素乘以2，使b的范数不断增大
        b = b * 2  # 将b乘以2
    
    # 条件判断：检查b的所有元素之和是否大于0
    if b.sum() > 0:  # 如果b的和大于0
        # 如果和为正，则c等于b
        c = b  # c = b
    else:  # 如果b的和小于等于0
        # 如果和为负或零，则c等于b的100倍
        c = 100 * b  # c = 100 * b
    
    # 返回计算后的张量c
    return c  # 返回c


a = torch.randn(size=(), requires_grad=True)  # 创建一个标量随机张量，需要计算梯度
d = f(a)  # 计算d = f(a)
d.backward()  # 计算d对a的梯度

print("a.grad:", a.grad)  # 打印a的梯度
print("a.grad == d / a:", a.grad == d / a)  # 打印比较结果

print("-"*20)  # 打印分隔线
