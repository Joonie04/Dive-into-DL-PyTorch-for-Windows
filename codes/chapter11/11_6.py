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

def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction="none")
    animator = Animator(xlabel="epoch", ylabel="loss", xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter), (evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f"loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch")

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

show_trace_2d(f_2d, train_2d(gd_2d))

eta = 0.6
show_trace_2d(f_2d, train_2d(gd_2d))

def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
show_trace_2d(f_2d, train_2d(momentum_2d))

eta, beta = 0.6, 0.25
show_trace_2d(f_2d, train_2d(momentum_2d))

setfigsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = torch.arange(40).detach().numpy()
    plt.plot(x, (1-beta)**x, label=f"beta = {beta:.2f}")
plt.xlabel("time")
plt.legend()
plt.show()

def init_momentum_states(feature_dim):
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams["momentum"] * v + p.grad
            p[:] -= hyperparams["lr"] * v
        p.grad.data.zero_()

def train_momentum(lr, momentum, num_epochs=2):
    train_ch11(sgd_momentum, init_momentum_states, {"lr": lr, "momentum": momentum}, data_iter, feature_dim, num_epochs)

data_iter, feature_dim = get_data_ch11(batch_size=10)

train_momentum(0.02, 0.5)
train_momentum(0.005, 0.9)
train_momentum(0.005, 0.9)

trainer = torch.optim.SGD
train_concise_ch11(trainer, {"lr": 0.005, "momentum": 0.9}, data_iter)

lambdas = [0.1, 1, 10, 19]
eta = 0.1
setfigsize((6, 4))
for lam in lambdas:
    t = torch.arange(20).detach().numpy()
    plt.plot(t, (1-eta*lam)**t, label=f"lambda = {lam:.2f}")
plt.xlabel("time")
plt.legend()
plt.show()
