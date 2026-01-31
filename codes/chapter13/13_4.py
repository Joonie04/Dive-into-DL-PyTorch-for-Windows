import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

torch.set_printoptions(2)

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

def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比
    
    参数:
        boxes1: 第一个边界框集合，形状为 (N, 4)
        boxes2: 第二个边界框集合，形状为 (M, 4)
    
    返回:
        IoU 矩阵，形状为 (N, M)，其中元素 [i, j] 表示 boxes1[i] 和 boxes2[j] 的 IoU
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
        ground_truth: 真实边界框，形状为 (num_gt, 5)，每行格式为 (class_id, x1, y1, x2, y2)
        anchors: 锚框，形状为 (num_anchors, 4)
        device: 设备
        iou_threshold: IoU 阈值，用于决定是否分配真实边界框
    
    返回:
        分配结果，形状为 (num_anchors,)，每个元素表示对应锚框分配的真实边界框索引，-1 表示背景
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
        anchors: 锚框，形状为 (N, 4)，格式为 (x1, y1, x2, y2)
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
        bbox_offset: 边界框偏移量
        bbox_mask: 边界框掩码
        class_labels: 类别标签
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
        每行格式为 (class_id, x1, y1, x2, y2, confidence)
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

if __name__ == "__main__":
    img_path = "dataset/img/catdog.jpg"
    if os.path.exists(img_path):
        img = Image.open(img_path)
    else:
        print(f"图像文件不存在: {img_path}")
        print("请将图像文件放置在 dataset/img/ 目录下")
        exit(1)
    
    h, w = img.size[1], img.size[0]
    print(f"image shape(h x w): {h} x {w}")
    
    X = torch.rand(size=(1, 3, h, w))
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print(f"Y shape: {Y.shape}")
    
    boxes = Y.reshape(h, w, 5, 4)
    print(f"boxes[250, 250, 0, :]: {boxes[250, 250, 0, :]}")
    
    set_figsize()
    bbox_scale = torch.tensor((w, h, w, h))
    fig = plt.imshow(img)
    show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ["s=0.75, r=1", "s=0.5, r=1", "s=0.25, r=1", "s=0.75, r=2",
                 "s=0.75, r=0.5"])
    
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                 [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                            [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                            [0.57, 0.3, 0.7, 0.5]])
    
    fig = plt.imshow(img)
    show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ["dog", "cat"], 'k')
    show_bboxes(fig.axes, anchors * bbox_scale, ["0", "1", "2", "3", "4"])
    
    labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
    print("labels[2]:", labels[2])
    print("labels[1]:", labels[1])
    print("labels[0]:", labels[0])
    
    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                            [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.8, 0.8]])
    offset_preds = torch.tensor([0] * anchors.numel())
    cls_probs = torch.tensor([[0] * 4, [0.9, 0.8, 0.7, 0.1], [0.1, 0.2, 0.3, 0.9]])
    
    fig = plt.imshow(img)
    show_bboxes(fig.axes, anchors * bbox_scale, ["dog=0.9", "dog=0.8", "dog=0.7", "dog=0.6"])
    
    output = multibox_detection(cls_probs.unsqueeze(dim=0),
                                offset_preds.unsqueeze(dim=0),
                                anchors.unsqueeze(dim=0),
                                nms_threshold=0.5)
    print("output", output)
    
    fig = plt.imshow(img)
    for i in output[0].detach().numpy():
        if i[-1] == -1: continue
        label = ("dog" if i[0] == 0 else "cat") + f"={i[-1]:.2f}"
        show_bboxes(fig.axes, [torch.tensor(i[1:5]) * bbox_scale], label)
