# 11.3 梯度下降
import numpy as np
import torch
import matplotlib.pyplot as plt

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

def contour(x, y, f, **kwargs):
    """绘制等高线图"""
    use_svg_display()
    axes = plt.gca()
    axes.contour(x, y, f, **kwargs)

def xlabel(text):
    """设置x轴标签"""
    plt.gca().set_xlabel(text)

def ylabel(text):
    """设置y轴标签"""
    plt.gca().set_ylabel(text)

# ========== 原始代码 ==========

## 11.3.1 一维梯度下降
def f(x):
    """目标函数"""
    return x ** 2

def f_grad(x):
    """目标函数的梯度"""
    return 2 * x

def gd(eta, f_grad):
    """梯度下降算法"""
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(x)
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
print("results", results)

def show_trace(results, f):
    """显示优化轨迹"""
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    set_figsize()
    plot([f_line, results], [[f(x) for x in f_line], [f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)

### 1. 学习率
show_trace(gd(0.05, f_grad), f)
show_trace(gd(1.1, f_grad), f)

### 2. 局部最小值
c = torch.tensor(0.15 * np.pi)

def f(x):
    """目标函数"""
    return x * torch.cos(c * x)

def f_grad(x):
    """目标函数的梯度"""
    return torch.cos(c * x) - c * x * torch.sin(c * x)

show_trace(gd(2, f_grad), f)


## 11.3.2 多元梯度下降
def train_2d(trainer, steps=20, f_grad=None):
    """训练2D函数"""
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):
    """显示2D优化轨迹"""
    set_figsize()
    plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1), torch.arange(-3.0, 1.0, 0.1))
    contour(x1, x2, f(x1, x2), colors='#1f77b4')
    xlabel('x1')
    ylabel('x2')

def f_2d(x1, x2):
    """2D目标函数"""
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):
    """2D目标函数的梯度"""
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    """2D梯度下降"""
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
# epochs 20, x1: -0.057646, x2: -0.000073



## 11.3.3 自适应方法
### 1. 牛顿法
c = torch.tensor(0.5)

def f(x):
    """目标函数"""
    return torch.cos(c * x)

def f_grad(x):
    """目标函数的梯度"""
    return c * torch.sinh(c * x)

def f_hess(x):
    """目标函数的海森矩阵"""
    return c ** 2 * torch.cos(c * x)

def newton(eta=1):
    """牛顿法"""
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(x)
    print(f'epoch 10, x: {x:f}')
    return results

show_trace(newton(), f)

c = torch.tensor(0.15 * np.pi)

def f(x):
    """目标函数"""
    return x * torch.cos(c * x)

def f_grad(x):
    """目标函数的梯度"""
    return torch.cos(c * x) - c * x * torch.sin(c * x)

def f_hess(x):
    """目标函数的海森矩阵"""
    return - 2 ** c * torch.sin(c * x) - x * c**2 * torch.cos(c * x)

show_trace(newton(), f)

show_trace(newton(0.5), f)
