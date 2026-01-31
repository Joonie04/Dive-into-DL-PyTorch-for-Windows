# 3.1 线性回归
## 3.1.2 向量化加速
import math  # 导入数学库
import time  # 导入时间库
import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库

n = 10000  # 定义向量长度
a = torch.ones(n)  # 创建一个长度为n的全1张量a
b = torch.ones(n)  # 创建一个长度为n的全1张量b

class Timer:  # 定义计时器类
    def __init__(self):  # 初始化方法
        self.times = []  # 存储所有计时结果
        self.start()  # 开始计时

    def start(self):  # 开始计时方法
        self.tik = time.time()  # 记录开始时间

    def stop(self):  # 停止计时方法
        self.times.append(time.time() - self.tik)  # 计算并存储耗时
        return self.times[-1]  # 返回最后一次耗时

    def avg(self):  # 计算平均耗时方法
        return sum(self.times) / len(self.times)  # 返回平均耗时

    def sum(self):  # 计算总耗时方法
        return sum(self.times)  # 返回总耗时

    def cumsum(self):  # 计算累积耗时方法
        return np.array(self.times).cumsum().tolist()  # 返回累积耗时列表

c = torch.zeros(n)  # 创建一个长度为n的全0张量c

timer = Timer()  # 创建计时器实例
for i in range(n):  # 循环n次
    c[i] = a[i] + b[i]  # 逐元素相加

print(f'{timer.stop():.5f} sec')  # 打印循环相加的耗时
timer.start()  # 重新开始计时
d = a + b  # 向量化相加
print(f'{timer.stop():.5f} sec')  # 打印向量化相加的耗时

## 3.1.3 正态分布
def normal(x, mu, sigma):  # 定义正态分布概率密度函数
    p = 1 / math.sqrt(2 * math.pi * sigma**2)  # 计算归一化常数
    return p * math.exp(-0.5 / sigma**2 * (x - mu)**2)  # 返回概率密度值

x = np.arange(-7, 7, 0.01)  # 创建从-7到7，步长为0.01的数组

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]  # 定义三组参数：(均值, 标准差)
plt.figure(figsize=(4.5, 2.5))  # 设置图形大小
for (mu, sigma), label in zip(params, [f'mean {mu}, std {sigma}' for mu, sigma in params]):  # 遍历参数和标签
    plt.plot(x, normal(x, mu, sigma), label=label)  # 绘制正态分布曲线
plt.xlabel('x')  # 设置x轴标签
plt.ylabel('p(x)')  # 设置y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图形