# 2.2 数据预处理

# 2.2.1 读取数据集
import os  # 导入操作系统模块
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset")  # 获取项目根目录下的dataset子目录路径

import pandas as pd  # 导入pandas库并简写为pd

data = pd.read_csv(os.path.join(data_dir, "house_tiny.csv"))  # 读取CSV文件到DataFrame
print("data:", data)  # 打印数据集


# 2.2.2 处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 3]  # 将数据集分为输入特征（前两列）和输出标签（第4列，LotArea）
inputs = inputs.fillna(inputs.mean())  # 用每列的平均值填充缺失值
print("处理缺失值后的inputs:", inputs)  # 打印处理后的输入数据
print("outputs:", outputs)  # 打印输出数据

# 2.2.3 转换为张量格式
import torch  # 导入PyTorch库

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values, dtype=torch.float32)  # 将DataFrame转换为PyTorch张量
print("X:", X)  # 打印输入特征张量
print("y:", y)  # 打印输出标签张量

