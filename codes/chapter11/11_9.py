import torch
from torch import nn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codes.chapter11.eleven_five import get_data_ch11, Timer, Animator, linreg, squared_loss, evaluate_loss, load_array, setfigsize, plot

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

def init_adadelta_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    delta_w = torch.zeros((feature_dim, 1))
    delta_b = torch.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams["rho"], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g**2
        p.grad.data.zero_()

data_iter, feature_dim = get_data_ch11(batch_size=10)
train_ch11(adadelta, init_adadelta_states, {"rho": 0.9}, data_iter, feature_dim)

trainer = torch.optim.Adadelta
train_ch11(trainer, lambda feature_dim: None, {"rho": 0.9}, data_iter, feature_dim)
