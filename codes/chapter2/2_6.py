# 2.6 概率

import torch  # 导入PyTorch库
from torch.distributions import multinomial  # 导入多项式分布
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库

fair_probs = torch.ones([6]) / 6  # 创建一个包含6个元素的均匀概率分布（每个面概率为1/6）
counts = multinomial.Multinomial(10, fair_probs).sample()  # 进行10次投掷，统计每个面出现的次数
print('counts:', counts)  # 打印计数结果

counts = multinomial.Multinomial(10, fair_probs).sample()  # 再次进行10次投掷
print('counts:', counts)  # 打印计数结果

counts = multinomial.Multinomial(1000, fair_probs).sample()  # 进行1000次投掷
counts /= 1000  # 将计数转换为频率
print('counts:', counts)  # 打印频率结果

print(""*20)  # 打印空字符串（分隔线）

counts = multinomial.Multinomial(10, fair_probs).sample((500,))  # 进行500组实验，每组10次投掷
print('counts:', counts)  # 打印计数结果
cum_counts = counts.cumsum(dim=0)  # 计算累积计数
print('cum_counts:', cum_counts)  # 打印累积计数

estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)  # 计算累积频率估计
print('estimates:', estimates)  # 打印频率估计

plt.rcParams['figure.figsize'] = (6, 6)  # 设置图形大小为6x6英寸
for i in range(6):  # 遍历6个面
    plt.plot(estimates[:, i].numpy(),  # 绘制第i个面的频率估计曲线
             label=("P(die=" + str(i + 1) + ")"))  # 设置图例标签
plt.axhline(y=1/6, color='black', linestyle='--')  # 绘制理论概率1/6的水平虚线
plt.legend()  # 显示图例
plt.show()  # 显示图形

