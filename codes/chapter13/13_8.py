# 13.8 区域卷积神经网络(R-CNN)系类

## 13.8.1 R-CNN
## 13.8.2 Fast R-CNN

import torch
import torchvision

X = torch.arange(16.).reshape((1, 1, 4, 4))
print("X:", X)

rois = torch.tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
print("rois:", rois)

torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
print("roi_pool:", torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1))

## 13.8.3 Faster R-CNN
## 13.8.4 Mask R-CNN

if __name__ == "__main__":
    print("=" * 60)
    print("演示 ROI Pooling 操作")
    print("=" * 60)
    
    X = torch.arange(16.).reshape((1, 1, 4, 4))
    print("\n输入特征图 X (batch=1, channel=1, height=4, width=4):")
    print(X)
    
    rois = torch.tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
    print("\n感兴趣区域 (Regions of Interest, ROIs):")
    print("rois 格式: [batch_index, x1, y1, x2, y2]")
    print("rois:", rois)
    print("  ROI 0: batch=0, (x1=0, y1=0, x2=20, y2=20)")
    print("  ROI 1: batch=0, (x1=10, y1=30, x2=30, y2=30)")
    
    output = torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
    print("\nROI Pooling 输出:")
    print("参数说明:")
    print("  - output_size=(2, 2): 每个ROI被池化为 2x2 的特征图")
    print("  - spatial_scale=0.1: 将ROI坐标从输入图像空间缩放到特征图空间")
    print("    (即特征图尺寸 = 图像尺寸 * spatial_scale)")
    print("\nroi_pool 输出形状:", output.shape)
    print("roi_pool 输出值:")
    print(output)
