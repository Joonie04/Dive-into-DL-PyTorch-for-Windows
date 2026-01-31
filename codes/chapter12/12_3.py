import torch
import time

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')

def try_all_gpus():
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(torch.device(f'cuda:{i}'))
    return devices if devices else [torch.device('cpu')]

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

devices = try_all_gpus()

def run(x):
    return [x * x for _ in range(50)]

x_gpu1 = torch.randn(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.randn(size=(4000, 4000), device=devices[1])

run(x_gpu1)
run(x_gpu2)
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with Benchmark("GPU1 time"):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with Benchmark("GPU2 time"):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])

def copy_to_cpu(x, non_blocking=False):
    return [y.to("cpu", non_blocking=non_blocking) for y in x]

with Benchmark("在GPU1上运行"):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with Benchmark("复制到CPU"):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()

with Benchmark("在GPU1上运行+复制到CPU"):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, non_blocking=True)
    torch.cuda.synchronize()
