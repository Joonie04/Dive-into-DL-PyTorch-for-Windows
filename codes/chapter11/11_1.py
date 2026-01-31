# 11.1 优化和深度学习
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

## 11.1.1 优化的目标
def f(x):
    """目标函数"""
    return x * torch.cos(np.pi * x)

def g(x):
    """带噪声的目标函数"""
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)

def annotate(text, xy, xytext):
    """在图中添加注释"""
    plt.gca().annotate(text, xy=xy, xytext=xytext,
                       arrowprops=dict(arrowstyle='->'))

x = torch.arange(0.5, 1.5, 0.01)
set_figsize((4.5, 2.5))
plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))


## 11.1.2 深度学习中的优化挑战
### 1. 局部极小值
x = torch.arange(-1.0, 2.0, 0.01)
plot(x, [f(x), g(x)], 'x', 'f(x)')
annotate('local min', (-0.3, -0.25), (-0.77, -1.0))
annotate('global min', (1.1, -0.95), (0.6, 0.8))

### 2. 鞍点
x = torch.arange(-2.0, 2.0, 0.01)
plot(x, x**3, 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
x, y = torch.meshgrid(torch.linspace(-1.0, 1.0, 101),
                      torch.linspace(-1.0, 1.0, 101))
z = x**2 - y**2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
plt.xticks(ticks)
plt.yticks(ticks)
ax.set_zticks(ticks)
plt.xlabel('x')
plt.ylabel('y')

### 3. 梯度消失
x = torch.arange(-2.0, 5.0, 0.01)
plot(x, [torch.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
