import os
import subprocess
import numpy
import torch
from torch import nn
import time

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')

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

class Benchmark:
    def __init__(self, description='Done'):
        self.description = description
    def __enter__(self):
        self.timer = Timer()
        return self
    def __exit__(self, *args):
        print(f"{self.description}: {self.timer.stop():.4f} sec")

device = try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with Benchmark("numpy"):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with Benchmark("torch"):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)

with Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)

x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
print("z", z)
