import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）
    
    参数:
        boxes: 边界框张量，形状为 (N, 4)，每行格式为 (x1, y1, x2, y2)
              x1, y1: 左上角坐标
              x2, y2: 右下角坐标
    
    返回:
        边界框张量，形状为 (N, 4)，每行格式为 (cx, cy, w, h)
        cx, cy: 中心点坐标
        w, h: 宽度和高度
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）
    
    参数:
        boxes: 边界框张量，形状为 (N, 4)，每行格式为 (cx, cy, w, h)
              cx, cy: 中心点坐标
              w, h: 宽度和高度
    
    返回:
        边界框张量，形状为 (N, 4)，每行格式为 (x1, y1, x2, y2)
        x1, y1: 左上角坐标
        x2, y2: 右下角坐标
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def load_image(url_or_path):
    """加载图像，支持 URL 或本地路径
    
    参数:
        url_or_path: 图像的 URL 或本地文件路径
    
    返回:
        PIL Image 对象
    """
    if url_or_path.startswith('http'):
        response = requests.get(url_or_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(url_or_path)
    return img

def bbox_to_rect(bbox, color):
    """将边界框(左上，右下)格式转换为matplotlib格式
    
    参数:
        bbox: 边界框，格式为 (x1, y1, x2, y2)
              x1, y1: 左上角坐标
              x2, y2: 右下角坐标
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

def show_image_with_bboxes(img, boxes, colors=None, labels=None):
    """显示图像并绘制边界框
    
    参数:
        img: PIL Image 对象
        boxes: 边界框列表，每个边界框格式为 (x1, y1, x2, y2)
        colors: 边界框颜色列表，如果为 None 则使用默认颜色
        labels: 边界框标签列表，如果为 None 则不显示标签
    """
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    if colors is None:
        colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange']
    
    for i, bbox in enumerate(boxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox, color)
        ax.add_patch(rect)
        
        if labels is not None and i < len(labels):
            ax.text(bbox[0], bbox[1] - 5, labels[i], 
                   bbox=dict(facecolor=color, alpha=0.5),
                   fontsize=12, color='white')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img_url = "http://d2l-data.s3-accelerate.amazonaws.com/catdog.jpg"
    img = load_image(img_url)
    
    dog_bbox = [60.0, 45.0, 378.0, 516.0]
    cat_bbox = [400.0, 112.0, 655.0, 493.0]
    
    boxes = torch.tensor((dog_bbox, cat_bbox))
    
    print("测试边界框转换函数:")
    print("box_center_to_corner(box_corner_to_center(boxes)) == boxes", 
          box_center_to_corner(box_corner_to_center(boxes)) == boxes)
    
    print("\n边界框信息:")
    print(f"狗的边界框 (左上, 右下): {dog_bbox}")
    print(f"猫的边界框 (左上, 右下): {cat_bbox}")
    
    center_boxes = box_corner_to_center(boxes)
    print(f"\n转换为中心格式 (中心x, 中心y, 宽度, 高度):")
    print(f"狗的边界框: {center_boxes[0].tolist()}")
    print(f"猫的边界框: {center_boxes[1].tolist()}")
    
    show_image_with_bboxes(img, [dog_bbox, cat_bbox], colors=['blue', 'red'], labels=['Dog', 'Cat'])
