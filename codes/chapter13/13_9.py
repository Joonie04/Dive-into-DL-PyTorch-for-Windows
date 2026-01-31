# 13.9 语义分割和数据集
## 13.9.1 图像分割和实例分割
## 13.9.2 Pascal VOC2012 语义分割数据集

import os
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def set_figsize(figsize=(3.5, 2.5)):
    """设置图像大小
    
    参数:
        figsize: 图像大小，格式为 (width, height)
    """
    plt.rcParams['figure.figsize'] = figsize

def show_images(images, num_rows, num_cols, scale=1.5):
    """显示多张图像
    
    参数:
        images: 图像列表或张量
        num_rows: 行数
        num_cols: 列数
        scale: 缩放比例
    
    返回:
        axes 对象列表
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() if num_rows * num_cols > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            if torch.is_tensor(img):
                img = img.permute(1, 2, 0).numpy()
            ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    return axes

def download_voc2012():
    """下载 Pascal VOC2012 数据集
    
    返回:
        数据集目录路径
    """
    dataset_dir = "dataset/voc2012"
    voc_dir = os.path.join(dataset_dir, "VOCdevkit", "VOC2012")
    
    if os.path.exists(voc_dir):
        print(f"数据集已存在于 {voc_dir}")
        return voc_dir
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from downloader.voc2012 import download_voc2012 as download
    return download()

voc_dir = download_voc2012()

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注
    
    参数:
        voc_dir: VOC数据集目录路径
        is_train: 是否为训练集，True 表示训练集，False 表示验证集
    
    返回:
        features: 图像列表，每个图像为张量
        labels: 标签列表，每个标签为张量
    """
    txt_fname = os.path.join(voc_dir, 'ImageSets/Segmentation/' + ('train.txt' if is_train else 'val.txt'))
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)

n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1, 2, 0) for img in imgs]
show_images(imgs, rows=2, cols=n)
plt.show()

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128], [128, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射
    
    返回:
        colormap2label: 形状为 (256^3,) 的张量，将RGB值映射到类别索引
    """
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引
    
    参数:
        colormap: 标签图像，形状为 (3, height, width)
        colormap2label: RGB到类别索引的映射
    
    返回:
        类别索引张量，形状为 (height, width)
    """
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

y = voc_label_indices(train_labels[0], voc_colormap2label())
print("y[105:115, 130:140]:", y[105:115, 130:140])
print("VOC_CLASSES[1]:", VOC_CLASSES[1])

### 1. 预处理数据
def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像
    
    参数:
        feature: 特征图像，形状为 (3, h, w)
        label: 标签图像，形状为 (3, h, w)
        height: 裁剪高度
        width: 裁剪宽度
    
    返回:
        (cropped_feature, cropped_label) 裁剪后的特征和标签
    """
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

imgs = []
for i in range(n):
    imgs += voc_rand_crop(train_features[i], train_labels[i], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
show_images(imgs[::2] + imgs[1::2], rows=2, cols=n)
plt.show()

### 2. 自定义语义分割数据集类
class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集
    
    参数:
        is_train: 是否为训练集
        crop_size: 裁剪尺寸 (height, width)
        voc_dir: VOC数据集目录路径
    """
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature) for feature in features]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        """归一化图像
        
        参数:
            img: 图像张量，形状为 (3, h, w)
        
        返回:
            归一化后的图像
        """
        return self.transform(img.float() / 255)

    def filter(self, labels):
        """过滤掉尺寸小于crop_size的标签
        
        参数:
            labels: 标签列表
        
        返回:
            过滤后的标签列表
        """
        return [label for label in labels if (
            label.shape[1] >= self.crop_size[0] and
            label.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        """获取单个样本
        
        参数:
            idx: 样本索引
        
        返回:
            (feature, label) 元组
            feature: 归一化后的特征图像
            label: 类别索引标签
        """
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],*self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        """返回数据集大小
        
        返回:
            数据集样本数量
        """
        return len(self.features)

### 3. 读取数据集
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)

batch_size = 64
num_workers = 0
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True, num_workers=num_workers)

for X, Y in train_iter:
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)
    break

### 4. 整合所有组件
def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集
    
    参数:
        batch_size: 批次大小
        crop_size: 裁剪尺寸 (height, width)
    
    返回:
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
    """
    voc_dir = download_voc2012()
    num_workers = 0
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        num_workers=num_workers)
    return train_iter, test_iter

if __name__ == "__main__":
    print("=" * 60)
    print("VOC2012 语义分割数据集")
    print("=" * 60)
    
    print("\n数据集类别:")
    for i, cls in enumerate(VOC_CLASSES):
        print(f"  {i:2d}: {cls}")
    
    print(f"\n数据集路径: {voc_dir}")
    print(f"训练集样本数: {len(voc_train)}")
    print(f"测试集样本数: {len(voc_test)}")
