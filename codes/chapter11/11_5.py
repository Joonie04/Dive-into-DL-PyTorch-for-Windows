import numpy as np
import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from downloader.airfoil import get_airfoil_dataset_path

class Timer:
    def __init__(self):
        self.times = []
        self.start()
    
    def start(self):
        self.tik = time.time()
    
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times) / len(self.times)
    
    def sum(self):
        return sum(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
    
    @property
    def elapsed(self):
        return self.times[-1] if self.times else 0

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.X, self.Y, self.fmts = None, None, fmts
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xlim, self.ylim = xlim, ylim
        self.xscale, self.yscale = xscale, yscale
        self.legend = legend
    
    def set_axes(self):
        for ax in self.axes:
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            ax.set_xscale(self.xscale)
            ax.set_yscale(self.yscale)
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            if self.legend:
                ax.legend(self.legend)
        self.X, self.Y = [], []
    
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.set_axes()
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.set_axes()
        plt.pause(0.001)
    
    @property
    def Y(self):
        return self._Y if hasattr(self, '_Y') else None
    
    @Y.setter
    def Y(self, value):
        self._Y = value

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def evaluate_loss(net, data_iter, loss):
    metric = torch.zeros(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric[0] += l.sum()
        metric[1] += l.numel()
    return metric[0] / metric[1]

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def setfigsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize

def plot(*args, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
         figsize=(3.5, 2.5)):
    if legend is None:
        legend = []
    setfigsize(figsize)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows * ncols == 1:
        axes = [axes, ]
    for i, (x, y) in enumerate(zip(args[0], args[1])):
        axes[0].plot(x, y, fmts[i % len(fmts)])
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_xscale(xscale)
    axes[0].set_yscale(yscale)
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    if legend:
        axes[0].legend(legend)
    plt.show()

timer = Timer()
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)

timer.start()
for i in range(256):
    C[i] = torch.mv(B, A[i])
timer.stop()
print("C:", C)
print("time:", timer.elapsed())

timer.start()
A = torch.mm(B, C)
timer.stop()
print("A:", A)
print("time:", timer.elapsed())

gigaflops = [2/i for i in timer.times]
print(f"performance in Gigaflops: element-wise {gigaflops[0]:.3f}, column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}")

timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print("A:", A)
print("time:", timer.elapsed())
print(f"gigaflops:, {2/timer.times[3]:.3f}")

def get_data_ch11(batch_size=10, n=1500):
    data_path = get_airfoil_dataset_path()
    data = np.genfromtxt(data_path, dtype=np.float32, delimiter="\t")
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1

def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams["lr"] * p.grad)
        p.grad.data.zero_()
    
def train_ch11(trainer_fn, states, hyperparams, data_iter, feature_dim, num_epochs=2):
    w = torch.normal(0.0, 0.01, size=(feature_dim, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    net, loss = lambda X: linreg(X, w, b), squared_loss
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
    return timer.cumsum(), animator.Y[0]

def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(sgd, None, {"lr": lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
sgd_res = train_sgd(0.005, 1)

mini1_res = train_sgd(.4, 100)
mini2_res = train_sgd(.05, 10)
setfigsize([6, 3])
plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
     'time(sec)', 'loss', xlim=[1e-2, 10],
     legend=["gd", "sgd", "batch size=100", "batch size=10"])
plt.gca().set_xscale("log")

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

data_iter, _ = get_data_ch11(batch_size=10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {"lr": 0.01}, data_iter)
