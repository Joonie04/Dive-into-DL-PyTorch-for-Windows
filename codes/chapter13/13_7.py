import os
import sys
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

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

def try_gpu(i=0):
    """尝试获取 GPU 设备
    
    参数:
        i: GPU 索引
    
    返回:
        如果 GPU 可用返回 GPU 设备，否则返回 CPU 设备
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')

class Accumulator:
    """累加器类，用于累加多个指标"""
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    """计时器类，用于测量代码执行时间"""
    def __init__(self):
        self.times = []
        self.start()
    
    def start(self):
        self.tik = time.time()
    
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times) / len(self.times)
    
    def sum(self):
        return sum(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

class Animator:
    """动画可视化类，用于绘制训练过程中的指标变化"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: self.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    
    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()
    
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.pause(0.001)
    
    @property
    def Y(self):
        return self._Y if hasattr(self, '_Y') else None
    
    @Y.setter
    def Y(self, value):
        self._Y = value

def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）
    
    参数:
        boxes: 边界框张量，形状为 (N, 4)，每行格式为 (x1, y1, x2, y2)
    
    返回:
        边界框张量，形状为 (N, 4)，每行格式为 (cx, cy, w, h)
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
    
    返回:
        边界框张量，形状为 (N, 4)，每行格式为 (x1, y1, x2, y2)
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

def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比
    
    参数:
        boxes1: 第一个边界框集合，形状为 (N, 4)
        boxes2: 第二个边界框集合，形状为 (M, 4)
    
    返回:
        IoU 矩阵，形状为 (N, M)
    """
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                               (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    
    inter_upperlefts = torch.max(boxes1[:, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    
    inter_areas = inters[:, 0] * inters[:, 1]
    union_areas = areas1 + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框
    
    参数:
        ground_truth: 真实边界框，形状为 (num_gt, 5)
        anchors: 锚框，形状为 (num_anchors, 4)
        device: 设备
        iou_threshold: IoU 阈值
    
    返回:
        分配结果，形状为 (num_anchors,)
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth[:, 1:])
    
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = max_idx % num_gt_boxes
        anc_idx = max_idx // num_gt_boxes
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bboxes, eps=1e-6):
    """对锚框偏移量的转换
    
    参数:
        anchors: 锚框，形状为 (N, 4)
        assigned_bboxes: 分配的真实边界框，形状为 (N, 4)
        eps: 防止除零的小常数
    
    返回:
        偏移量，形状为 (N, 4)
    """
    c_anc = box_corner_to_center(anchors)
    c_assigned_bboxes = box_corner_to_center(assigned_bboxes)
    offset_xy = (c_assigned_bboxes[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = torch.log(c_assigned_bboxes[:, 2:] / c_anc[:, 2:] + eps)
    offset = torch.cat((offset_xy, offset_wh), dim=1)
    return offset

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框
    
    参数:
        anchors: 锚框，形状为 (1, num_anchors, 4)
        labels: 真实标签，形状为 (batch_size, num_gt, 5)
    
    返回:
        (bbox_offset, bbox_mask, class_labels)
    """
    batch_size, anchors = anchors.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_classes = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bboxes = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long()
        assigned_bboxes[indices_true] = label[bb_idx, 1:]
        offset = offset_boxes(anchors, assigned_bboxes) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框
    
    参数:
        anchors: 锚框，形状为 (N, 4)
        offset_preds: 预测的偏移量，形状为 (N, 4)
    
    返回:
        预测的边界框，形状为 (N, 4)
    """
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 2) + (anc[:, :2] + offset_preds[:, 2:]) / 2
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 2) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), dim=1)
    return pred_bbox

def nms(boxes, scores, iou_threshold):
    """非极大值抑制
    
    参数:
        boxes: 边界框，形状为 (N, 4)
        scores: 置信度分数，形状为 (N,)
        iou_threshold: IoU 阈值
    
    返回:
        保留的边界框索引
    """
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=scores.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框
    
    参数:
        cls_probs: 类别概率，形状为 (batch_size, num_classes, num_anchors)
        offset_preds: 偏移量预测，形状为 (batch_size, num_anchors, 4)
        anchors: 锚框，形状为 (1, num_anchors, 4)
        nms_threshold: NMS 的 IoU 阈值
        pos_threshold: 正样本的置信度阈值
    
    返回:
        预测结果，形状为 (batch_size, num_anchors, 5)
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bbox = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bbox, conf, nms_threshold)
        
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bbox = conf[all_id_sorted], predicted_bbox[all_id_sorted]
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_bbox = torch.cat((predicted_bbox, conf.unsqueeze(1)), dim=1)
        out.append(pred_bbox)
    return torch.stack(out)

def cls_predictor(num_inputs, num_anchors, num_classes):
    """类别预测层
    
    参数:
        num_inputs: 输入通道数
        num_anchors: 每个位置的锚框数量
        num_classes: 类别数量
    
    返回:
        卷积层，输出形状为 (batch, num_anchors * (num_classes + 1), h, w)
    """
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                      kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    """边界框预测层
    
    参数:
        num_inputs: 输入通道数
        num_anchors: 每个位置的锚框数量
    
    返回:
        卷积层，输出形状为 (batch, num_anchors * 4, h, w)
    """
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

def forward(x, block):
    """前向传播
    
    参数:
        x: 输入张量
        block: 网络块
    
    返回:
        输出张量
    """
    return block(x)

def flatten_pred(pred):
    """展平预测结果
    
    参数:
        pred: 预测张量，形状为 (batch, channels, h, w)
    
    返回:
        展平后的张量，形状为 (batch, h * w * channels)
    """
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    """连接多尺度预测结果
    
    参数:
        preds: 预测列表
    
    返回:
        连接后的张量
    """
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

def down_sample_blk(num_inputs, num_outputs):
    """高和宽减半块
    
    参数:
        num_inputs: 输入通道数
        num_outputs: 输出通道数
    
    返回:
        下采样块
    """
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(num_inputs, num_outputs, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(num_outputs))
        blk.append(nn.ReLU())
        num_inputs = num_outputs
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

def base_net():
    """基础网络块
    
    返回:
        基础网络
    """
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

def get_blk(i):
    """获取指定索引的网络块
    
    参数:
        i: 块索引
    
    返回:
        网络块
    """
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    """块前向传播
    
    参数:
        X: 输入张量
        blk: 网络块
        size: 锚框大小列表
        ratio: 锚框宽高比列表
        cls_predictor: 类别预测器
        bbox_predictor: 边界框预测器
    
    返回:
        (Y, anchors, cls_preds, bbox_preds)
    """
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    """TinySSD 模型
    
    参数:
        num_classes: 类别数量
    """
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        """前向传播
        
        参数:
            X: 输入张量，形状为 (batch, 3, 256, 256)
        
        返回:
            (anchors, cls_preds, bbox_preds)
        """
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    """计算损失函数
    
    参数:
        cls_preds: 类别预测
        cls_labels: 类别标签
        bbox_preds: 边界框预测
        bbox_labels: 边界框标签
        bbox_masks: 边界框掩码
    
    返回:
        总损失
    """
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    """评估类别预测准确率
    
    参数:
        cls_preds: 类别预测
        cls_labels: 类别标签
    
    返回:
        正确预测的数量
    """
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    """评估边界框预测误差
    
    参数:
        bbox_preds: 边界框预测
        bbox_labels: 边界框标签
        bbox_masks: 边界框掩码
    
    返回:
        绝对误差的总和
    """
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

def predict(X, net, device):
    """预测目标
    
    参数:
        X: 输入图像
        net: 模型
        device: 设备
    
    返回:
        预测结果
    """
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = cls_preds.softmax(dim=-1).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

def display(img, output, threshold=0.5):
    """显示预测结果
    
    参数:
        img: 图像
        output: 预测输出
        threshold: 置信度阈值
    """
    set_figsize((5, 5))
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

if __name__ == "__main__":
    print("测试类别预测层和边界框预测层...")
    Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
    Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
    print("Y1.shape", Y1.shape)
    print("Y2.shape", Y2.shape)
    
    print("\n测试预测结果连接...")
    print("concat_preds([Y1, Y2]).shape", concat_preds([Y1, Y2]).shape)
    
    print("\n测试下采样块...")
    print("forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape", 
          forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)
    
    print("\n测试基础网络...")
    print("base_net()(torch.zeros((2, 3, 256, 256))).shape", 
          base_net()(torch.zeros((2, 3, 256, 256))).shape)
    
    print("\n测试 TinySSD 模型...")
    net = TinySSD(num_classes=1)
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)
    print("anchors.shape", anchors.shape)
    print("cls_preds.shape", cls_preds.shape)
    print("bbox_preds.shape", bbox_preds.shape)
    
    print("\n训练模型...")
    batch_size = 32
    train_iter, _ = load_data_bananas(batch_size)
    device = try_gpu()
    net = TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')
    
    num_epochs, timer = 20, Timer()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
    net = net.to(device)
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        net.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                       bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        animator.add(epoch + 1, (cls_err, bbox_mae))
        print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
        print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
              f'{str(device)}')
    
    print("\n预测目标...")
    img_path = "dataset/img/banana.jpg"
    if os.path.exists(img_path):
        X = torchvision.io.read_image(img_path).unsqueeze(0).float()
        img = X.squeeze(0).permute(1, 2, 0).long()
        output = predict(X, net, device)
        display(img, output.cpu(), threshold=0.9)
    else:
        print(f"图像文件不存在: {img_path}")
        print("请将图像文件放置在 dataset/img/ 目录下")
