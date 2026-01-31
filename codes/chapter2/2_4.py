# 2.4 微积分

import numpy as np  # 导入NumPy库
from matplotlib_inline import backend_inline  # 导入matplotlib_inline后端
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库

def f(x):  # 定义函数f(x)
    return 3*x**2 - 4*x  # 返回3x² - 4x

def numerical_lim(f, x, h):  # 定义数值极限计算函数
    return (f(x+h) - f(x)) / h  # 计算差分商

h = 0.1  # 初始步长
for i in range(5):  # 循环5次
    print(f"h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}")  # 打印步长和数值极限
    h *= 0.1  # 步长缩小10倍

def use_svg_display():  # 定义使用SVG显示的函数
    backend_inline.set_matplotlib_formats('svg')  # 设置matplotlib使用SVG格式

def set_figsize(figsize=(3.5, 2.5)):  # 定义设置图形大小的函数
    use_svg_display()  # 使用SVG显示
    plt.rcParams['figure.figsize'] = figsize  # 设置图形大小

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):  # 定义设置坐标轴的函数
    axes.set_xlabel(xlabel)  # 设置x轴标签
    axes.set_ylabel(ylabel)  # 设置y轴标签
    axes.set_xscale(xscale)  # 设置x轴刻度类型
    axes.set_yscale(yscale)  # 设置y轴刻度类型
    axes.set_xlim(xlim)  # 设置x轴范围
    axes.set_ylim(ylim)  # 设置y轴范围
    if legend:  # 如果有图例
        axes.legend(legend)  # 添加图例
    axes.grid()  # 显示网格

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):  # 定义绘图函数
    if legend is None:  # 如果图例为None
        legend = []  # 初始化为空列表
    set_figsize(figsize)  # 设置图形大小
    axes = axes if axes else plt.gca()  # 如果没有指定坐标轴，则获取当前坐标轴
    def has_one_axis(X):  # 定义检查是否为一维数据的函数
        return (hasattr(X, "ndim") and X.ndim == 1 or  # 检查是否为一维NumPy数组
                isinstance(X, list) and not hasattr(X[0], "__len__"))  # 检查是否为元素不是列表的列表
    if has_one_axis(X):  # 如果X是一维数据
        X = [X]  # 转换为列表
    if Y is None:  # 如果Y为None
        X, Y = [[]] * len(X), X  # 交换X和Y
    elif has_one_axis(Y):  # 如果Y是一维数据
        Y = [Y]  # 转换为列表
    if len(X) != len(Y):  # 如果X和Y的长度不同
        X = X * len(Y)  # 复制X以匹配Y的长度
    axes.cla()  # 清除坐标轴
    for x, y, fmt in zip(X, Y, fmts):  # 遍历X、Y和格式
        if len(x):  # 如果x不为空
            axes.plot(x, y, fmt)  # 绘制x和y
        else:  # 如果x为空
            axes.plot(y, fmt)  # 只绘制y
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)  # 设置坐标轴

x = np.arange(0, 3, 0.1)  # 创建从0到3，步长为0.1的数组
plot(x, [f(x), 2*x-3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])  # 绘制函数和切线
plt.show()  # 显示图形
