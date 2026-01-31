# 11.2 凸性
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========== 从10_2.py复制的绘图函数 ==========

def use_svg_display():
    """使用SVG格式显示图形"""
    plt.rcParams['figure.figsize'] = (3.5, 2.5)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

def set_figsize(figsize=(3.5, 2.5)):
    """设置图形大小"""
    use_svg_display()
    plt.figure(figsize=figsize)

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []
    use_svg_display()
    if axes is None:
        axes = plt.gca()
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [], X
    if has_one_axis(Y):
        Y = [Y]
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def has_one_axis(X):
    """如果X只有一个轴，则返回True"""
    return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
            and not hasattr(X[0], "__len__"))

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置坐标轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

# ========== 原始代码 ==========

## 11.2.1 定义
### 1. 凸集
### 2. 凸函数
f = lambda x: 0.5 * x**2
g = lambda x: torch.cos(np.pi * x)
h = lambda x: torch.exp(0.5 * x)

x, segment = torch.arange(-2.0, 2.0, 0.01), torch.tensor([-1.5, 1])
use_svg_display()
_, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    plot([x, segment], [func(x), func(segment)], axes=ax)

### 3. 詹森不等式


## 11.2.2 性质
### 1. 局部极小值是全局极小值
f = lambda x: (x - 1) ** 2
set_figsize()
plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
