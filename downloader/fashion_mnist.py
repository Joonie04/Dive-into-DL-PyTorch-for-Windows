import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

def download_fashion_mnist():
    """下载Fashion-MNIST数据集到指定目录"""
    
    # 设置下载目录
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("开始下载Fashion-MNIST数据集...")
    
    # 下载训练集
    print("下载训练集...")
    trainset = torchvision.datasets.FashionMNIST(
        root=str(dataset_dir),
        train=True,
        download=True,
        transform=transform
    )
    
    # 下载测试集
    print("下载测试集...")
    testset = torchvision.datasets.FashionMNIST(
        root=str(dataset_dir),
        train=False,
        download=True,
        transform=transform
    )
    
    print("数据集下载完成！")
    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    print(f"数据集类别: {trainset.classes}")
    print(f"数据存储路径: {dataset_dir.absolute()}")
    
    return trainset, testset

if __name__ == "__main__":
    download_fashion_mnist()