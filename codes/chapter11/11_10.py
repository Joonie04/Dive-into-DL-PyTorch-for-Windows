import torch
from torch import nn
from torch.utils import data
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from downloader.airfoil import get_airfoil_dataset_path
from codes.chapter11.eleven_five import Timer, Animator, linreg, squared_loss, evaluate_loss, load_array

def get_data_ch11(batch_size=10, n=1500):
    data_path = get_airfoil_dataset_path()
    data = np.genfromtxt(data_path, dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1

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

def init_adam_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return ((s_w, v_w), (s_b, v_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (s, v) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            p[:] -= hyperparams["lr"] * v / (torch.sqrt(s) + eps)
        p.grad.data.zero_()

data_iter, feature_dim = get_data_ch11(batch_size=10)
train_ch11(adam, init_adam_states, {"lr": 0.01, "t": 1}, data_iter, feature_dim)

trainer = torch.optim.Adam
train_ch11(trainer, lambda feature_dim: None, {"lr": 0.01}, data_iter, feature_dim)

def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (s, v) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.square(p.grad) * torch.sign(torch.square(p.grad) - s)
            p[:] -= hyperparams["lr"] * v / (torch.sqrt(s) + eps)
        p.grad.data.zero_()
    hyperparams["t"] += 1

data_iter, feature_dim = get_data_ch11(batch_size=10)
train_ch11(yogi, init_adam_states, {"lr": 0.01, "t": 1}, data_iter, feature_dim)
