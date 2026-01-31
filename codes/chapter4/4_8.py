import torch  # 导入PyTorch库
from matplotlib import pyplot as plt  # 导入matplotlib.pyplot绘图模块

# 绘制图表
def plot(X, Y, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):  # 定义绘制数据点的函数
    """绘制数据点"""
    if legend is None:  # 如果没有提供图例
        legend = []  # 初始化空列表
    
    set_figsize(figsize)  # 设置图表大小
    axes = axes if axes else plt.gca()  # 使用传入的axes或获取当前axes
    
    # 检查X和Y是否为列表或数组
    def has_one_axis(X):  # 定义检查X是否为一维的函数
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))  # 返回是否为一维
    
    if has_one_axis(X):  # 如果X是一维的
        X = [X] * len(Y)  # 为每个Y复制X
    
    axes.cla()  # 清除axes
    for x, y, fmt in zip(X, Y, fmts):  # 遍历X、Y和格式
        if len(x):  # 如果x非空
            axes.plot(x, y, fmt)  # 绘制曲线
        else:  # 如果x为空
            axes.plot(y, fmt)  # 只绘制y
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)  # 设置坐标轴

# 设置图表大小
def set_figsize(figsize=(3.5, 2.5)):  # 定义设置matplotlib图表大小的函数
    """设置matplotlib的图表大小"""
    plt.rcParams['figure.figsize'] = figsize  # 设置图表大小

# 设置图表坐标轴
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):  # 定义设置matplotlib坐标轴的函数
    """设置matplotlib的坐标轴"""
    axes.set_xlabel(xlabel)  # 设置x轴标签
    axes.set_ylabel(ylabel)  # 设置y轴标签
    axes.set_xscale(xscale)  # 设置x轴缩放
    axes.set_yscale(yscale)  # 设置y轴缩放
    axes.set_xlim(xlim)  # 设置x轴范围
    axes.set_ylim(ylim)  # 设置y轴范围
    if legend:  # 如果有图例
        axes.legend(legend)  # 显示图例
    axes.grid()  # 显示网格

# 梯度消失
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)  # 创建x张量，范围从-8到8，步长0.1
y = torch.sigmoid(x)  # 计算sigmoid函数值
y.backward(torch.ones_like(x))  # 反向传播计算梯度

plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],  # 绘制sigmoid函数和其梯度
     legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))  # 设置图例和图表大小
plt.show()  # 显示图形

# 梯度爆炸
M = torch.normal(0, 1, size=(4, 4))  # 生成4x4的正态分布矩阵
print('一个矩阵', M)  # 打印初始矩阵
for i in range(100):  # 迭代100次
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))  # 矩阵乘法
print('100次迭代后的矩阵', M)  # 打印迭代后的矩阵
