import math
import torch
import torchvision
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import time

dataset_path = 'dataset/FashionMNIST/raw'

def load_data_fashion_mnist(batch_size, resize=None):
    trans = []
    if resize:
        trans.append(transforms.Resize(size=resize))
    trans.append(transforms.ToTensor())
    
    transform = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=dataset_path, train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=dataset_path, train=False, download=False, transform=transform)
    
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_iter, test_iter

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(torch.float32).sum())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: self.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    
    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()
    
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.pause(0.001)
    
    @property
    def Y(self):
        return self._Y if hasattr(self, '_Y') else None
    
    @Y.setter
    def Y(self, value):
        self._Y = value

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))
    return model

loss = nn.CrossEntropyLoss()
device = try_gpu()

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

def train(net, train_iter, test_iter, loss, num_epochs, device, lr_scheduler_fn=None):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metrics = Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metrics.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_loss, train_acc = metrics[0] / metrics[2], metrics[1] / metrics[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter), (train_loss, train_acc, None))

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                scheduler.step()
            else:
                for param_group in scheduler.param_groups:
                    param_group["lr"] = scheduler.get_last_lr()[0]

        print(f"epoch {epoch + 1}, train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")

lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
scheduler = None
train(net, train_iter, test_iter, loss, num_epochs, device)

lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f"learning rate: {trainer.param_groups[0]['lr']:.2f}")

class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, epoch):
        return self.lr * pow(epoch + 1.0, -0.5)

scheduler = SquareRootScheduler(lr=0.1)
plt.figure(figsize=(6, 4))
plt.plot(torch.arange(num_epochs), [scheduler(i) for i in range(num_epochs)])
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.title('Square Root Scheduler')
plt.grid(True, alpha=0.3)
plt.show()

net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, loss, num_epochs, device, scheduler)

class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, epoch):
        self.lr = self.base_lr * self.factor**epoch
        return max(self.stop_factor_lr, self.lr)

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
plt.figure(figsize=(6, 4))
plt.plot(torch.arange(50), [scheduler(i) for i in range(50)])
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.title('Factor Scheduler')
plt.grid(True, alpha=0.3)
plt.show()

net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

plt.figure(figsize=(6, 4))
plt.plot(torch.arange(num_epochs), [get_lr(trainer, scheduler) for i in range(num_epochs)])
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.title('Multi-Step LR Scheduler')
plt.grid(True, alpha=0.3)
plt.show()

net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)
train(net, train_iter, test_iter, loss, num_epochs, device, scheduler)

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
                 warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=num_epochs, base_lr=0.3, final_lr=0.01)
plt.figure(figsize=(6, 4))
plt.plot(torch.arange(num_epochs), [scheduler(i) for i in range(num_epochs)])
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.title('Cosine Scheduler')
plt.grid(True, alpha=0.3)
plt.show()

net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
scheduler = CosineScheduler(max_update=num_epochs, base_lr=0.3, final_lr=0.01)
train(net, train_iter, test_iter, loss, num_epochs, device, scheduler)

scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
plt.figure(figsize=(6, 4))
plt.plot(torch.arange(num_epochs), [scheduler(i) for i in range(num_epochs)])
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.title('Cosine Scheduler with Warmup')
plt.grid(True, alpha=0.3)
plt.show()

net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
train(net, train_iter, test_iter, loss, num_epochs, device, scheduler)
