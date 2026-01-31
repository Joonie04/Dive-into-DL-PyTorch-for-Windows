import math
import torch
import matplotlib.pyplot as plt
import numpy as np

def f(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):
    return (2 * x1, 4 * x2)

def sgd(x1, x2, s1, s2, f_grad, eta=0.1):
    g1, g2 = f_grad(x1, x2)
    g1 += torch.normal(0.0, 1, (1,))
    g2 += torch.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

def constant_lr():
    return 1

def train_2d(trainer, steps=20, f_grad=None):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    return results

def show_trace_2d(f, results):
    plt.figure(figsize=(6, 4))
    x1_vals = np.linspace(-5.5, 3.5, 100)
    x2_vals = np.linspace(-3.0, 3.0, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f(X1, X2)
    
    cp = plt.contour(X1, X2, Z, levels=20)
    plt.clabel(cp, inline=True, fontsize=8)
    
    results = np.array(results)
    plt.plot(results[:, 0], results[:, 1], '-o', color='red', markersize=4)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Optimization trajectory for f(x1,x2) = x1² + 2x2²')
    plt.grid(True, alpha=0.3)
    plt.show()

eta = 0.1
lr = constant_lr
results = train_2d(sgd, steps=50, f_grad=f_grad)
show_trace_2d(f, results)

def exponential_lr():
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
results = train_2d(sgd, steps=1000, f_grad=f_grad)
show_trace_2d(f, results)

def polynomial_lr():
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
results = train_2d(sgd, steps=50, f_grad=f_grad)
show_trace_2d(f, results)
