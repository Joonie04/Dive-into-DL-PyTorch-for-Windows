import torch
from torch import nn
import time

def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print("fancy_func(1, 2, 3, 4)", fancy_func(1, 2, 3, 4))

def add_():
    return '''
    def add(a, b):
        return a + b
    '''

def fancy_func_():
    return '''
    def fancy_func(a, b, c, d):
        e = add(a, b)
        f = add(c, d)
        g = add(e, f)
        return g
    '''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print("prog", prog)
y = compile(prog, '', 'exec')
exec(y)

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
        import numpy as np
        return np.array(self.times).cumsum().tolist()

def get_net():
    net = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    return net

x = torch.randn(size=(1, 512))
net = get_net()
print("net(x)", net(x))

net = torch.jit.script(net)
print("net(x)", net(x))

class Benchmark:
    def __init__(self, description='Done'):
        self.description = description
    def __enter__(self):
        self.timer = Timer()
        return self
    def __exit__(self, *args):
        print(f"{self.description}: {self.timer.stop():.4f} sec")

net = get_net()
with Benchmark("Eager mode"):
    for _ in range(1000):
        net(x)
net = torch.jit.script(net)
with Benchmark("TorchScript mode"):
    for _ in range(1000):
        net(x)

net.save("my_mlp")
