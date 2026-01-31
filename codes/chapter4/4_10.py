import numpy as np  # 导入NumPy库
import pandas as pd  # 导入Pandas库
import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch神经网络模块
from matplotlib import pyplot as plt  # 导入matplotlib.pyplot绘图模块
import sys  # 导入系统模块
import os  # 导入操作系统模块

# 设置数据集路径
dataset_path = 'dataset/kaggle_house'  # 设置Kaggle房价预测数据集路径

# 加载数据集
train_data = pd.read_csv(f'{dataset_path}/train.csv')  # 读取训练数据
test_data = pd.read_csv(f'{dataset_path}/test.csv')  # 读取测试数据

print(train_data.shape)  # 打印训练数据的形状
print(test_data.shape)  # 打印测试数据的形状

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])  # 打印训练数据的前4行的特定列

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))  # 合并训练和测试数据的特征

# 若无法获得测试数据, 可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  # 获取数值型特征的索引
all_features[numeric_features] = all_features[numeric_features].apply(  # 对数值型特征进行标准化
    lambda x: (x - x.mean()) / (x.std()))  # 标准化：减去均值，除以标准差
# 在标准化数据之后, 所有均值消失, 因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)  # 将缺失值填充为0

# dummy_na将所有缺失值替换为"None"
all_features = pd.get_dummies(all_features, dummy_na=True)  # 对分类特征进行独热编码
print(all_features.shape)  # 打印特征矩阵的形状

# 确保所有特征都是数值类型
print("Checking feature dtypes...")  # 打印检查信息
non_numeric_cols = all_features.select_dtypes(include=['object']).columns  # 获取非数值型列
if len(non_numeric_cols) > 0:  # 如果存在非数值型列
    print(f"Found non-numeric columns: {non_numeric_cols}")  # 打印非数值型列
    # 将非数值列转换为数值类型
    all_features = all_features.apply(pd.to_numeric, errors='coerce')  # 转换为数值类型
    # 填充转换过程中产生的缺失值
    all_features = all_features.fillna(0)  # 填充缺失值为0

n_train = train_data.shape[0]  # 获取训练样本数
# 转换为numpy数组并确保所有值都是float32类型
train_features_np = all_features[:n_train].values.astype(np.float32)  # 转换训练特征为float32
test_features_np = all_features[n_train:].values.astype(np.float32)  # 转换测试特征为float32

# 转换为PyTorch张量
train_features = torch.tensor(train_features_np, dtype=torch.float32)  # 转换为PyTorch张量
test_features = torch.tensor(test_features_np, dtype=torch.float32)  # 转换为PyTorch张量
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)  # 转换标签为PyTorch张量

loss = nn.MSELoss()  # 定义均方误差损失函数
in_features = train_features.shape[1]  # 获取输入特征维度

# 加载数据
def load_array(data_arrays, batch_size, is_train=True):  # 定义构造PyTorch数据迭代器的函数
    """构造一个PyTorch数据迭代器"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)  # 创建数据集
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)  # 创建数据迭代器

# 绘制图表
def plot(X, Y, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):  # 定义绘制数据点的函数
    """绘制数据点"""
    if legend is None:  # 如果没有提供图例
        legend = []  # 初始化空列表
    
    set_figsize(figsize)  # 设置图表大小
    axes = axes if axes else plt.gca()  # 使用传入的axes或获取当前axes
    
    # 确保X和Y都是可迭代的
    if isinstance(X, (list, np.ndarray)) and not isinstance(X[0], (list, np.ndarray)):  # 如果X是一维的
        # X是一维的，为每个Y的子列表复制一个X
        X = [X] * len(Y)  # 为每个Y复制X
    
    axes.cla()  # 清除axes
    for i, (x, y) in enumerate(zip(X, Y)):  # 遍历X和Y
        fmt = fmts[i % len(fmts)]  # 获取对应的格式
        axes.plot(x, y, fmt)  # 绘制曲线
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)  # 设置坐标轴

# 设置图表大小
def set_figsize(figsize=(3.5, 2.5)):  # 定义设置matplotlib图表大小的函数
    """设置matplotlib的图表大小"""
    plt.rcParams['figure.figsize'] = figsize  # 设置图表大小

# 设置图表坐标轴
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):  # 定义设置matplotlib坐标轴的函数
    """设置matplotlib的坐标轴"""
    axes.set_xlabel(xlabel)  # 设置x轴标签
    axes.set_ylabel(ylabel)  # 设置y轴标签
    axes.set_xscale(xscale)  # 设置x轴缩放
    axes.set_yscale(yscale)  # 设置y轴缩放
    axes.set_xlim(xlim)  # 设置x轴范围
    axes.set_ylim(ylim)  # 设置y轴范围
    if legend:  # 如果有图例
        axes.legend(legend)  # 显示图例
    axes.grid()  # 显示网格

def get_net():  # 定义获取模型的函数
    net = nn.Sequential(nn.Linear(in_features, 1))  # 定义线性回归模型
    return net  # 返回模型

def log_rmse(net, features, labels):  # 定义计算对数均方根误差的函数
    # 为了在取对数时进一步稳定该值, 将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))  # 将预测值限制在[1, inf)范围内
    rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))  # 计算对数均方根误差
    return rmse.item()  # 返回标量值

def train(net, train_feautres, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):  # 定义训练函数
    train_ls, test_ls = [], []  # 初始化训练和测试损失列表
    train_iter = load_array((train_feautres, train_labels), batch_size)  # 创建训练数据迭代器
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 定义Adam优化器
    for epoch in range(num_epochs):  # 遍历每个epoch
        for X, y in train_iter:  # 遍历训练数据迭代器
            optimizer.zero_grad()  # 清零梯度
            l = loss(net(X), y)  # 计算损失
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
        train_ls.append(log_rmse(net, train_feautres, train_labels))  # 记录训练损失
        if test_labels is not None:  # 如果有测试标签
            test_ls.append(log_rmse(net, test_features, test_labels))  # 记录测试损失
    return train_ls, test_ls  # 返回训练和测试损失

def get_k_fold_data(k, i, X, y):  # 定义获取K折交叉验证数据的函数
    assert k > 1  # 确保折数大于1
    fold_size = X.shape[0] // k  # 计算每折的大小
    X_train, y_train = None, None  # 初始化训练数据和标签
    for j in range(k):  # 遍历每一折
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 获取当前折的索引
        X_part, y_part = X[idx, :], y[idx]  # 获取当前折的数据
        if j == i:  # 如果是验证折
            X_valid, y_valid = X_part, y_part  # 设置验证数据和标签
        elif X_train is None:  # 如果训练数据为空
            X_train, y_train = X_part, y_part  # 设置训练数据和标签
        else:  # 如果训练数据不为空
            X_train = torch.cat([X_train, X_part], 0)  # 拼接训练数据
            y_train = torch.cat([y_train, y_part], 0)  # 拼接训练标签
    return X_train, y_train, X_valid, y_valid  # 返回训练和验证数据

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):  # 定义K折交叉验证函数
    train_l_sum, valid_l_sum = 0, 0  # 初始化训练和验证损失总和
    for i in range(k):  # 遍历每一折
        data = get_k_fold_data(k, i, X_train, y_train)  # 获取K折数据
        net = get_net()  # 创建模型
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)  # 训练模型
        train_l_sum += train_ls[-1]  # 累加训练损失
        valid_l_sum += valid_ls[-1]  # 累加验证损失
        if i == 0:  # 如果是第一折
            plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],  # 绘制训练和验证损失曲线
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],  # 设置坐标轴
                     legend=['train', 'valid'], yscale='log')  # 设置图例和y轴缩放
            plt.show()  # 显示图形
        # 打印当前折的训练和验证损失
        print(f'折{i + 1}, 训练log rmse{float(train_ls[-1]):f}, '  
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k  # 返回平均训练和验证损失

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64  # 设置K折交叉验证的超参数
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)  # 执行K折交叉验证
# 打印平均训练和验证损失
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):  # 定义训练和预测函数
    net = get_net()  # 创建模型
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)  # 训练模型
    plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='log rmse', xlim=[1, num_epochs], yscale='log')  # 绘制训练损失曲线
    plt.show()  # 显示图形
    print(f'训练log rmse: {float(train_ls[-1]):f}')  # 打印最终训练损失
    # 将网络应用于测试集
    preds = net(test_features).detach().numpy()  # 对测试集进行预测
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])  # 将预测结果添加到测试数据
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)  # 创建提交文件
    submission.to_csv('submission.csv', index=False)  # 保存为CSV文件

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)  # 执行训练和预测
