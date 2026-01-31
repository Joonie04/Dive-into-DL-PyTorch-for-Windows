import os
import sys
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def bbox_to_rect(bbox, color):
    """将边界框(左上，右下)格式转换为matplotlib格式
    
    参数:
        bbox: 边界框，格式为 (x1, y1, x2, y2)
        color: 边界框颜色
    
    返回:
        matplotlib Rectangle 对象
    """
    return patches.Rectangle(
        xy=(bbox[0], bbox[1]), 
        width=bbox[2]-bbox[0], 
        height=bbox[3]-bbox[1],
        fill=False, 
        edgecolor=color, 
        linewidth=2
    )

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """在图像上显示边界框
    
    参数:
        axes: matplotlib 的 axes 对象
        bboxes: 边界框列表，每个边界框格式为 (x1, y1, x2, y2)
        labels: 边界框标签列表
        colors: 边界框颜色列表
    """
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    
    labels = _make_list(labels)
    colors = _make_list(colors, ["b", "g", "r", "m", "c"])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        if torch.is_tensor(bbox):
            bbox = bbox.detach().numpy()
        rect = bbox_to_rect(bbox, color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = "k" if color == "w" else "w"
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va="center", ha="center", fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def read_data_bananas(is_train=True, data_dir=None):
    """读取香蕉检测数据集中的图像和标签
    
    参数:
        is_train: 是否为训练集，True 表示训练集，False 表示验证集
        data_dir: 数据集目录路径
    
    返回:
        images: 图像列表，每个图像为张量
        targets: 标签张量，形状为 (num_samples, 1, 5)
                 每个标签格式为 (class_id, x1, y1, x2, y2)
    """
    if data_dir is None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
        from downloader.banana_detection import download_banana_detection
        data_dir = download_banana_detection()
        if data_dir is None:
            raise RuntimeError("数据集下载失败")
    
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 
                         'images', f'{img_name}')))
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananaDataset(torch.utils.data.Dataset):
    """香蕉检测数据集类
    
    参数:
        is_train: 是否为训练集
    """
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        """获取单个样本
        
        参数:
            idx: 样本索引
        
        返回:
            (image, label) 元组
            image: 图像张量，形状为 (3, 256, 256)
            label: 标签张量，形状为 (1, 5)
        """
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        """返回数据集大小
        
        返回:
            数据集样本数量
        """
        return len(self.features)

def load_data_bananas(batch_size):
    """加载香蕉检测数据集
    
    参数:
        batch_size: 批次大小
    
    返回:
        train_iter: 训练数据迭代器
        val_iter: 验证数据迭代器
    """
    train_iter = torch.utils.data.DataLoader(BananaDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananaDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

if __name__ == "__main__":
    batch_size, edge_size = 32, 256
    
    print("加载香蕉检测数据集...")
    train_iter, _ = load_data_bananas(batch_size)
    
    print("\n获取一个批次的数据...")
    batch = next(iter(train_iter))
    print("batch[0].shape (图像批次形状):", batch[0].shape)
    print("batch[1].shape (标签批次形状):", batch[1].shape)
    
    print("\n演示: 显示前 10 张图像及其边界框")
    imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
    axes = show_images(imgs, 2, 5)
    for ax, label in zip(axes, batch[1][0:10]):
        show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
    plt.show()
