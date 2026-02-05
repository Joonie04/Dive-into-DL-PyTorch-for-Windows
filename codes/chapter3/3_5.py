# 3.5 图像分类数据集
import torch  # 导入PyTorch库
import torchvision  # 导入torchvision库，用于计算机视觉任务
from torch.utils import data  # 导入PyTorch数据处理工具
from torchvision import transforms  # 导入图像变换模块

# 设置matplotlib后端为TkAgg，以便在命令行环境中显示图像
import matplotlib  # 导入matplotlib库
matplotlib.use('TkAgg')  # 设置matplotlib使用TkAgg后端
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot绘图模块

# 设置matplotlib使用SVG显示
plt.rcParams['figure.figsize'] = (3.5, 2.5)  # 设置图形大小为3.5x2.5英寸
plt.rcParams['figure.dpi'] = 100  # 设置图形分辨率为100 DPI

def get_fashion_mnist_labels(labels):  # 定义获取Fashion-MNIST标签文本的函数
    """返回Fashion-MNIST数据集的文本标签"""  # 函数文档字符串
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',  # 定义10个类别的文本标签
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]  # 返回标签对应的文本列表

def show_images(imgs, num_rows, num_cols, title=None, scale=1.5):  # 定义显示图像的函数
    """绘制图像列表"""  # 函数文档字符串
    figsize = (num_cols * scale, num_rows * scale)  # 计算图形大小
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)  # 创建子图网格
    axes = axes.flatten()  # 将子图数组展平为一维
    for i, (ax, img) in enumerate(zip(axes, imgs)):  # 遍历每个子图和图像
        if torch.is_tensor(img):  # 如果图像是PyTorch张量
            # 图片张量
            ax.imshow(img.numpy())  # 将张量转换为NumPy数组并显示
        else:  # 如果图像不是张量
            # PIL图片
            ax.imshow(img)  # 直接显示PIL图像
        ax.axes.get_xaxis().set_visible(False)  # 隐藏x轴
        ax.axes.get_yaxis().set_visible(False)  # 隐藏y轴
        if title:  # 如果有标题
            ax.set_title(title[i])  # 设置子图标题
    return axes  # 返回子图对象


if __name__ == '__main__':  # 当脚本作为主程序运行时执行

    # 使用SVG显示已在文件开头设置
    
    # 读取数据
    trans = transforms.ToTensor()  # 定义图像转换为张量的变换
    data_path = 'dataset/FashionMNIST/raw'  # 设置数据集存储路径
    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=trans, download=False)  # 加载训练集
    test_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=trans, download=False)  # 加载测试集
    
    # 检查数据是否加载成功
    print(f"训练集样本数: {len(train_dataset)}")  # 打印训练集样本数量
    print(f"测试集样本数: {len(test_dataset)}")  # 打印测试集样本数量

    # 查看训练集第一个样本的图像形状
    print(f"训练集第一个样本的图像形状: {train_dataset[0][0].shape}")  # 打印第一个样本的形状

    X, y = next(iter(data.DataLoader(train_dataset, batch_size=18)))  # 获取一个批次的数据（18张图像）
    axes = show_images(X.reshape(18, 28, 28), 2, 9, title=get_fashion_mnist_labels(y))  # 显示18张图像，2行9列
    
    # 显示图像
    plt.show()  # 显示图形
