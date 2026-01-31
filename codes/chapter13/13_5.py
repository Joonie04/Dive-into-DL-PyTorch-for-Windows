import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
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

def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框
    
    参数:
        data: 输入数据张量，形状为 (batch_size, channels, height, width)
        sizes: 缩放比例列表，如 [0.75, 0.5, 0.25]
        ratios: 宽高比列表，如 [1, 2, 0.5]
    
    返回:
        锚框张量，形状为 (1, num_anchors, 4)
        num_anchors = height * width * (len(sizes) + len(ratios) - 1)
    """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width
    
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing="ij")
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                    sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                    sizes[0] / torch.sqrt(ratio_tensor[1:])))
    
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2
    
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

def set_figsize(figsize=(3.5, 2.5)):
    """设置图像大小
    
    参数:
        figsize: 图像大小，格式为 (width, height)
    """
    plt.rcParams['figure.figsize'] = figsize

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
    """显示所有边界框
    
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

def display_anchors(fmap_w, fmap_h, s, img, w, h):
    """显示不同尺度的锚框
    
    参数:
        fmap_w: 特征图宽度
        fmap_h: 特征图高度
        s: 缩放比例列表
        img: 图像
        w: 图像宽度
        h: 图像高度
    """
    set_figsize()
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    show_bboxes(plt.imshow(img).axes,
                anchors[0] * bbox_scale)

if __name__ == "__main__":
    img_path = "dataset/img/catdog.jpg"
    if os.path.exists(img_path):
        img = plt.imread(img_path)
    else:
        print(f"图像文件不存在: {img_path}")
        print("请将图像文件放置在 dataset/img/ 目录下")
        exit(1)
    
    h, w = img.shape[:2]
    print("h:", h, "w:", w)
    
    print("\n显示 4x4 特征图的锚框 (缩放比例 0.15):")
    print("较小的锚框适合检测小对象")
    display_anchors(fmap_w=4, fmap_h=4, s=[0.15], img=img, w=w, h=h)
    
    print("\n显示 2x2 特征图的锚框 (缩放比例 0.4):")
    print("中等大小的锚框适合检测中等大小的对象")
    display_anchors(fmap_w=2, fmap_h=2, s=[0.4], img=img, w=w, h=h)
    
    print("\n显示 1x1 特征图的锚框 (缩放比例 0.8):")
    print("较大的锚框适合检测大对象")
    display_anchors(fmap_w=1, fmap_h=1, s=[0.8], img=img, w=w, h=h)
