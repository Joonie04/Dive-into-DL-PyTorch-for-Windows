# 2.7 查阅文档

# 2.7.1 查找模块中的所有函数和类
import torch
print(dir(torch.distributions))

# 2.7.2 查找特定函数和类的用法
print(help(torch.ones))
print("torch.ones(4):", torch.ones(4))