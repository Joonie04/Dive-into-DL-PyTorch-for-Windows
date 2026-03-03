# 4.1 多层感知机
## 4.1.2 激活函数
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5, 2.5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_activation(x, y, title, xlabel='x', ylabel='y'):
    plt.figure(figsize=(5, 2.5))
    plt.plot(x.detach().numpy(), y.detach().numpy(), 'b-', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_grad(x, grad, title, xlabel='x', ylabel='grad'):
    plt.figure(figsize=(5, 2.5))
    plt.plot(x.detach().numpy(), grad.detach().numpy(), 'r-', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

### 1. ReLU函数
print("=" * 50)
print("1. ReLU函数 (Rectified Linear Unit)")
print("   公式: ReLU(x) = max(0, x)")
print("=" * 50)
y = torch.relu(x)
plot_activation(x, y, 'ReLU(x)')
y.backward(torch.ones_like(x), retain_graph=True)
plot_grad(x, x.grad, 'grad of ReLU')
x.grad.zero_()

### 2. Sigmoid函数
print("\n" + "=" * 50)
print("2. Sigmoid函数")
print("   公式: sigmoid(x) = 1 / (1 + exp(-x))")
print("=" * 50)
y = torch.sigmoid(x)
plot_activation(x, y, 'Sigmoid(x)')
y.backward(torch.ones_like(x), retain_graph=True)
plot_grad(x, x.grad, 'grad of Sigmoid')
x.grad.zero_()

### 3. Tanh函数
print("\n" + "=" * 50)
print("3. Tanh函数 (双曲正切)")
print("   公式: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))")
print("=" * 50)
y = torch.tanh(x)
plot_activation(x, y, 'Tanh(x)')
y.backward(torch.ones_like(x), retain_graph=True)
plot_grad(x, x.grad, 'grad of Tanh')
x.grad.zero_()

### 4. LeakyReLU函数
print("\n" + "=" * 50)
print("4. LeakyReLU函数")
print("   公式: LeakyReLU(x) = x if x > 0 else alpha*x")
print("=" * 50)
leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)
y = leaky_relu(x)
plot_activation(x, y, 'LeakyReLU(x) (alpha=0.1)')
y.backward(torch.ones_like(x), retain_graph=True)
plot_grad(x, x.grad, 'grad of LeakyReLU')
x.grad.zero_()

### 5. ELU函数
print("\n" + "=" * 50)
print("5. ELU函数 (Exponential Linear Unit)")
print("   公式: ELU(x) = x if x > 0 else alpha*(exp(x)-1)")
print("=" * 50)
elu = torch.nn.ELU(alpha=1.0)
y = elu(x)
plot_activation(x, y, 'ELU(x) (alpha=1.0)')
y.backward(torch.ones_like(x), retain_graph=True)
plot_grad(x, x.grad, 'grad of ELU')
x.grad.zero_()

### 6. GELU函数
print("\n" + "=" * 50)
print("6. GELU函数 (Gaussian Error Linear Unit)")
print("   公式: GELU(x) = x * Phi(x), Phi为标准正态分布CDF")
print("=" * 50)
y = F.gelu(x)
plot_activation(x, y, 'GELU(x)')
y.backward(torch.ones_like(x), retain_graph=True)
plot_grad(x, x.grad, 'grad of GELU')
x.grad.zero_()

### 7. SiLU/Swish函数
print("\n" + "=" * 50)
print("7. SiLU函数 (Swish)")
print("   公式: SiLU(x) = x * sigmoid(x)")
print("=" * 50)
y = F.silu(x)
plot_activation(x, y, 'SiLU/Swish(x)')
y.backward(torch.ones_like(x), retain_graph=True)
plot_grad(x, x.grad, 'grad of SiLU')
x.grad.zero_()

### 8. Softplus函数
print("\n" + "=" * 50)
print("8. Softplus函数")
print("   公式: Softplus(x) = log(1 + exp(x))")
print("=" * 50)
y = F.softplus(x)
plot_activation(x, y, 'Softplus(x)')
y.backward(torch.ones_like(x), retain_graph=True)
plot_grad(x, x.grad, 'grad of Softplus')
x.grad.zero_()

### 9. Mish函数
print("\n" + "=" * 50)
print("9. Mish函数")
print("   公式: Mish(x) = x * tanh(softplus(x))")
print("=" * 50)
softplus_x = F.softplus(x)
y = x * torch.tanh(softplus_x)
plot_activation(x, y, 'Mish(x)')
y.backward(torch.ones_like(x), retain_graph=True)
plot_grad(x, x.grad, 'grad of Mish')
x.grad.zero_()

### 10. Hardswish函数
print("\n" + "=" * 50)
print("10. Hardswish函数")
print("   公式: Hardswish(x) = x * ReLU6(x+3)/6")
print("=" * 50)
y = F.hardswish(x)
plot_activation(x, y, 'Hardswish(x)')
y.backward(torch.ones_like(x), retain_graph=True)
plot_grad(x, x.grad, 'grad of Hardswish')
x.grad.zero_()

### 11. 激活函数对比图
print("\n" + "=" * 50)
print("11. 激活函数对比")
print("=" * 50)
x_compare = torch.arange(-4.0, 4.0, 0.01)
activations = {
    'ReLU': torch.relu(x_compare),
    'Sigmoid': torch.sigmoid(x_compare),
    'Tanh': torch.tanh(x_compare),
    'LeakyReLU': torch.nn.LeakyReLU(0.1)(x_compare),
    'ELU': torch.nn.ELU()(x_compare),
    'GELU': F.gelu(x_compare),
    'SiLU': F.silu(x_compare),
    'Softplus': F.softplus(x_compare),
}

plt.figure(figsize=(10, 6))
for name, y in activations.items():
    plt.plot(x_compare.numpy(), y.detach().numpy(), label=name, linewidth=1.5)
plt.xlabel('x')
plt.ylabel('Activation(x)')
plt.title('Comparison of Activation Functions')
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim(-4, 4)
plt.ylim(-2, 4)
plt.show()

print("\n所有激活函数演示完成！")
