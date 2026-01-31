import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codes.chapter11.eleven_five import get_data_ch11, Timer, Animator, linreg, squared_loss, evaluate_loss, load_array, setfigsize, plot

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
    plt.title(f'Optimization trajectory for f(x1,x2)')
    plt.grid(True, alpha=0.3)
    plt.show()

def train_ch11(trainer_fn, init_states_fn, hyperparams, data_iter, feature_dim, num_epochs=2):
    w = torch.normal(0.0, 0.01, size=(feature_dim, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    net, loss = lambda X: linreg(X, w, b), squared_loss
    states = init_states_fn(feature_dim)
    animator = Animator(xlabel="epoch", ylabel="loss", xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter), (evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f"loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch")

setfigsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = torch.arange(40).detach().numpy()
    plt.plot(x, (1 - gamma) * gamma**x, label=f'gamma = {gamma:.2f}')
plt.xlabel('time')
plt.legend()
plt.show()

def rmsprop_2d(x1, x2, s1, s2, f_grad):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 0.4 * x2
    s1 = gamma * s1 + (1 - gamma) * g1**2
    s2 = gamma * s2 + (1 - gamma) * g2**2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1**2 + 2 * x2**2

eta, gamma = 0.4, 0.9
show_trace_2d(f_2d, train_2d(rmsprop_2d))

def init_rmsprop_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    eps = 1e-6
    gamma = hyperparams["gamma"]
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams["lr"] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()

data_iter, feature_dim = get_data_ch11(batch_size=10)
train_ch11(rmsprop, init_rmsprop_states, {"lr": 0.1, "gamma": 0.9}, data_iter, feature_dim)

trainer = torch.optim.RMSprop
train_ch11(trainer, lambda feature_dim: None, {"lr": 0.1, "alpha": 0.9}, data_iter, feature_dim)
