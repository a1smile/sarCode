# -*- coding: utf-8 -*-
"""
SAR图像多类别有向目标检测工具函数
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 数据预处理工具

def normalize_intensity(image, min_val=0, max_val=255):
    """
    归一化图像强度到指定范围
    """
    if np.max(image) - np.min(image) < 1e-8:
        return image
    normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    normalized = normalized * (max_val - min_val) + min_val
    return normalized.astype(np.uint8)

def adaptive_histogram_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    自适应直方图均衡化，增强图像对比度
    """
    if len(image.shape) == 3:
        # 彩色图像需要转换为YCrCb格式
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        # 灰度图像直接处理
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)

def add_speckle_noise(image, mean=0, var=0.01):
    """
    添加斑点噪声（SAR图像特有噪声）
    """
    row, col = image.shape[:2]
    gauss = np.random.normal(mean, var**0.5, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + image * gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# 几何变换工具

def rotate_image(image, angle, center=None, scale=1.0):
    """
    旋转图像
    """
    (h, w) = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated

def resize_image(image, size, keep_aspect_ratio=False, pad_mode=cv2.BORDER_CONSTANT, pad_value=0):
    """
    调整图像大小，可选保持宽高比
    """
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        scale = min(size[0] / h, size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # 计算填充
        pad_h = (size[0] - new_h) // 2
        pad_w = (size[1] - new_w) // 2
        pad_h2 = size[0] - new_h - pad_h
        pad_w2 = size[1] - new_w - pad_w
        
        # 填充图像
        padded = cv2.copyMakeBorder(resized, pad_h, pad_h2, pad_w, pad_w2, 
                                   pad_mode, value=pad_value)
        return padded
    else:
        return cv2.resize(image, (size[1], size[0]))

# 边界框处理工具

def xywh_to_xyxy(bbox):
    """
    将中心点+宽高格式的边界框转换为左上右下格式
    """
    x, y, w, h = bbox
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return [x1, y1, x2, y2]

def xyxy_to_xywh(bbox):
    """
    将左上右下格式的边界框转换为中心点+宽高格式
    """
    x1, y1, x2, y2 = bbox
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [x, y, w, h]

def rotate_bbox(bbox, angle, center=None):
    """
    旋转边界框
    """
    x, y, w, h = bbox
    
    if center is None:
        # 以边界框中心为旋转中心
        center = (x, y)
    
    # 计算边界框的四个顶点
    corners = np.array([
        [x - w/2, y - h/2],
        [x + w/2, y - h/2],
        [x + w/2, y + h/2],
        [x - w/2, y + h/2]
    ])
    
    # 构建旋转矩阵
    angle_rad = math.radians(angle)
    rotation_matrix = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad), math.cos(angle_rad)]
    ])
    
    # 旋转每个顶点
    rotated_corners = []
    for corner in corners:
        # 平移到原点
        translated = corner - center
        # 旋转
        rotated = np.dot(rotation_matrix, translated)
        # 平移回原位置
        rotated_corner = rotated + center
        rotated_corners.append(rotated_corner)
    
    rotated_corners = np.array(rotated_corners)
    
    # 计算旋转后的边界框（轴对齐边界框）
    min_x = np.min(rotated_corners[:, 0])
    min_y = np.min(rotated_corners[:, 1])
    max_x = np.max(rotated_corners[:, 0])
    max_y = np.max(rotated_corners[:, 1])
    
    # 返回中心点+宽高格式
    new_x = (min_x + max_x) / 2
    new_y = (min_y + max_y) / 2
    new_w = max_x - min_x
    new_h = max_y - min_y
    
    return [new_x, new_y, new_w, new_h]

# 角度处理工具

def angle_to_sin_cos(angle):
    """
    将角度转换为正弦和余弦表示
    """
    rad = math.radians(angle)
    return math.sin(rad), math.cos(rad)

def sin_cos_to_angle(sin_val, cos_val):
    """
    将正弦和余弦表示转换为角度
    """
    rad = math.atan2(sin_val, cos_val)
    return math.degrees(rad)

def normalize_angle(angle, min_angle=-90, max_angle=90):
    """
    将角度规范化到指定范围内
    """
    while angle < min_angle:
        angle += 360
    while angle > max_angle:
        angle -= 360
    return angle

def angle_difference(angle1, angle2):
    """
    计算两个角度之间的最小差值（考虑周期性）
    """
    diff = angle1 - angle2
    while diff < -180:
        diff += 360
    while diff > 180:
        diff -= 360
    return diff

# 旋转边界框IoU计算

def rotated_iou(box1, box2):
    """
    计算两个旋转边界框的IoU
    这里box格式为[x, y, w, h, angle]，其中angle为角度
    """
    # 创建旋转矩形
    rect1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
    rect2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
    
    # 获取旋转矩形的轮廓
    points1 = cv2.boxPoints(rect1)
    points2 = cv2.boxPoints(rect2)
    
    # 计算交集区域
    intersection = calculate_intersection_area(points1, points2)
    
    # 计算两个矩形的面积
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    
    # 计算IoU
    iou = intersection / (area1 + area2 - intersection + 1e-8)
    
    return iou

def calculate_intersection_area(points1, points2):
    """
    计算两个多边形的交集面积
    """
    # 创建掩码图像
    h = max(np.max(points1[:, 1]), np.max(points2[:, 1])) + 1
    w = max(np.max(points1[:, 0]), np.max(points2[:, 0])) + 1
    
    mask1 = np.zeros((int(h), int(w)), dtype=np.uint8)
    mask2 = np.zeros((int(h), int(w)), dtype=np.uint8)
    
    # 填充多边形
    cv2.fillPoly(mask1, [points1.astype(np.int32)], 1)
    cv2.fillPoly(mask2, [points2.astype(np.int32)], 1)
    
    # 计算交集
    intersection = cv2.bitwise_and(mask1, mask2)
    
    # 返回交集面积
    return np.sum(intersection)

# 旋转非极大值抑制

def rotated_nms(detections, iou_threshold=0.5):
    """
    旋转非极大值抑制
    过滤重叠的检测结果
    detections格式: [x, y, w, h, angle, score, class_id]
    """
    # 按置信度排序
    if len(detections) == 0:
        return []
    
    # 转换为numpy数组
    detections = np.array(detections)
    
    # 按置信度排序
    sorted_indices = np.argsort(-detections[:, 5])
    detections = detections[sorted_indices]
    
    keep = []
    while len(detections) > 0:
        # 保留置信度最高的检测框
        current = detections[0]
        keep.append(current)
        
        # 计算与其他检测框的IoU
        if len(detections) > 1:
            ious = []
            for det in detections[1:]:
                iou = rotated_iou(current[:5], det[:5])
                ious.append(iou)
            ious = np.array(ious)
            
            # 保留IoU小于阈值的检测框
            detections = detections[1:][ious < iou_threshold]
        else:
            break
    
    return np.array(keep)

# 评估指标计算

def calculate_precision_recall(true_positives, false_positives, false_negatives):
    """
    计算精确率和召回率
    """
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    return precision, recall

def calculate_f1_score(precision, recall):
    """
    计算F1分数
    """
    return 2 * (precision * recall) / (precision + recall + 1e-8)

def calculate_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    计算平均精确率(AP)
    """
    # 按置信度排序
    sorted_indices = np.argsort(-predictions[:, 5])
    predictions = predictions[sorted_indices]
    
    # 初始化变量
    true_positives = np.zeros(len(predictions))
    false_positives = np.zeros(len(predictions))
    
    # 标记已匹配的真实边界框
    matched_gt = np.zeros(len(ground_truths))
    
    # 遍历所有预测边界框
    for i, pred in enumerate(predictions):
        # 找到与当前预测边界框IoU最高的真实边界框
        max_iou = -1
        max_idx = -1
        
        for j, gt in enumerate(ground_truths):
            # 只考虑同一类别的边界框
            if pred[6] == gt[6] and matched_gt[j] == 0:
                iou = rotated_iou(pred[:5], gt[:5])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
        
        # 如果IoU大于阈值，则视为真正例
        if max_iou >= iou_threshold:
            true_positives[i] = 1
            matched_gt[max_idx] = 1
        else:
            false_positives[i] = 1
    
    # 计算累积的精确率和召回率
    cumulative_true_positives = np.cumsum(true_positives)
    cumulative_false_positives = np.cumsum(false_positives)
    
    # 计算精确率和召回率
    precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives + 1e-8)
    recall = cumulative_true_positives / (len(ground_truths) + 1e-8)
    
    # 计算AP（使用11点插值法）
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap, precision, recall

def calculate_map(predictions_dict, ground_truths_dict, iou_threshold=0.5):
    """
    计算平均平均精确率(mAP)
    predictions_dict和ground_truths_dict是按类别ID组织的字典
    """
    aps = []
    
    for class_id in predictions_dict:
        if class_id not in ground_truths_dict or len(ground_truths_dict[class_id]) == 0:
            continue
        
        ap, _, _ = calculate_ap(predictions_dict[class_id], ground_truths_dict[class_id], iou_threshold)
        aps.append(ap)
    
    # 计算mAP
    if len(aps) == 0:
        return 0.0
    
    return np.mean(aps)

def calculate_confusion_matrix(predictions, ground_truths, num_classes):
    """
    计算混淆矩阵
    """
    y_true = []
    y_pred = []
    
    # 遍历所有真实边界框
    for gt in ground_truths:
        # 找到与当前真实边界框IoU最高的预测边界框
        max_iou = -1
        best_pred_class = -1
        
        for pred in predictions:
            iou = rotated_iou(gt[:5], pred[:5])
            if iou > max_iou:
                max_iou = iou
                best_pred_class = pred[6]
        
        # 如果找到匹配的预测边界框，则记录类别
        if max_iou >= 0.5:
            y_true.append(gt[6])
            y_pred.append(best_pred_class)
        else:
            y_true.append(gt[6])
            y_pred.append(num_classes)  # 视为背景类
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes + 1)))
    
    return cm

# 可视化工具

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, label=None):
    """
    在图像上绘制边界框
    bbox格式: [x, y, w, h, angle]（旋转边界框）或[x1, y1, x2, y2]（轴对齐边界框）
    """
    img_copy = image.copy()
    
    if len(bbox) == 5:
        # 旋转边界框
        x, y, w, h, angle = bbox
        rect = ((x, y), (w, h), angle)
        points = cv2.boxPoints(rect)
        points = np.int0(points)
        cv2.polylines(img_copy, [points], True, color, thickness)
        
        # 如果有标签，绘制标签
        if label is not None:
            # 找到最左上角的点作为标签位置
            min_idx = np.argmin(np.sum(points, axis=1))
            text_pos = tuple(points[min_idx])
            cv2.putText(img_copy, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    else:
        # 轴对齐边界框
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # 如果有标签，绘制标签
        if label is not None:
            cv2.putText(img_copy, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return img_copy

def visualize_detections(image, detections, class_names, score_threshold=0.5):
    """
    可视化检测结果
    detections格式: [x, y, w, h, angle, score, class_id]
    """
    img_copy = image.copy()
    
    # 遍历所有检测结果
    for det in detections:
        x, y, w, h, angle, score, class_id = det
        
        # 过滤低置信度的检测结果
        if score < score_threshold:
            continue
        
        # 为不同类别选择不同颜色
        color = ((class_id * 73) % 255, (class_id * 137) % 255, (class_id * 191) % 255)
        
        # 绘制边界框
        rect = ((x, y), (w, h), angle)
        points = cv2.boxPoints(rect)
        points = np.int0(points)
        cv2.polylines(img_copy, [points], True, color, 2)
        
        # 绘制标签
        label = f"{class_names[int(class_id)]}: {score:.2f}"
        # 找到最左上角的点作为标签位置
        min_idx = np.argmin(np.sum(points, axis=1))
        text_pos = tuple(points[min_idx])
        cv2.putText(img_copy, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img_copy

def plot_precision_recall_curve(precision, recall, ap, class_name):
    """
    绘制精确率-召回率曲线
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap:.3f}')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve for {class_name}')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    return plt

def plot_confusion_matrix(cm, class_names):
    """
    绘制混淆矩阵
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # 添加类别标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return plt

# 模型辅助工具

def load_model(model_path, model, device='cuda'):
    """
    加载模型权重
    """
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def save_model(model, optimizer, epoch, loss, model_path):
    """
    保存模型权重
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, model_path)

def freeze_layers(model, freeze_names):
    """
    冻结指定的层
    """
    for name, param in model.named_parameters():
        for freeze_name in freeze_names:
            if freeze_name in name:
                param.requires_grad = False
                break
    return model

def unfreeze_layers(model, unfreeze_names=None):
    """
    解冻指定的层，或解冻所有层
    """
    for name, param in model.named_parameters():
        if unfreeze_names is None:
            param.requires_grad = True
        else:
            for unfreeze_name in unfreeze_names:
                if unfreeze_name in name:
                    param.requires_grad = True
                    break
    return model

def count_parameters(model):
    """
    计算模型参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_flops(model, input_size=(3, 640, 640)):
    """
    估计模型的FLOPs
    需要安装thop库: pip install thop
    """
    try:
        from thop import profile
        input_tensor = torch.randn(1, *input_size)
        flops, params = profile(model, inputs=(input_tensor,))
        return flops, params
    except ImportError:
        print("请安装thop库以计算FLOPs: pip install thop")
        return None, None

# 数据增强工具

def mosaic_augmentation(images, annotations, img_size=640):
    """
    马赛克数据增强
    将4张图像拼接成一张大图
    """
    # 创建马赛克图像
    mosaic_img = np.zeros((img_size * 2, img_size * 2, 3), dtype=np.uint8)
    
    # 计算随机中心点
    xc, yc = np.random.randint(img_size // 2, img_size * 3 // 2, 2)
    
    # 存储转换后的标注
    mosaic_annotations = []
    
    # 遍历4张图像
    for i in range(4):
        img, ann = images[i], annotations[i]
        
        # 调整图像大小
        img = resize_image(img, (img_size, img_size))
        
        # 确定图像在马赛克中的位置
        if i == 0:  # 左上角
            x1a, y1a, x2a, y2a = 0, 0, xc, yc
            x1b, y1b, x2b, y2b = img_size - xc, img_size - yc, img_size, img_size
        elif i == 1:  # 右上角
            x1a, y1a, x2a, y2a = xc, 0, img_size * 2, yc
            x1b, y1b, x2b, y2b = 0, img_size - yc, img_size * 2 - xc, img_size
        elif i == 2:  # 左下角
            x1a, y1a, x2a, y2a = 0, yc, xc, img_size * 2
            x1b, y1b, x2b, y2b = img_size - xc, 0, img_size, img_size * 2 - yc
        else:  # 右下角
            x1a, y1a, x2a, y2a = xc, yc, img_size * 2, img_size * 2
            x1b, y1b, x2b, y2b = 0, 0, img_size * 2 - xc, img_size * 2 - yc
        
        # 放置图像
        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        
        # 转换标注
        for bbox in ann:
            # 假设bbox格式为[x, y, w, h, angle, class_id]
            x, y, w, h, angle, class_id = bbox
            
            # 转换坐标
            x = x * (x2b - x1b) / img_size + x1a
            y = y * (y2b - y1b) / img_size + y1a
            w = w * (x2b - x1b) / img_size
            h = h * (y2b - y1b) / img_size
            
            # 添加到马赛克标注
            mosaic_annotations.append([x, y, w, h, angle, class_id])
    
    # 调整马赛克图像大小
    mosaic_img = resize_image(mosaic_img, (img_size, img_size))
    
    # 调整标注坐标
    scale = img_size / (img_size * 2)
    for i in range(len(mosaic_annotations)):
        mosaic_annotations[i][0] *= scale  # x
        mosaic_annotations[i][1] *= scale  # y
        mosaic_annotations[i][2] *= scale  # w
        mosaic_annotations[i][3] *= scale  # h
    
    return mosaic_img, mosaic_annotations

def large_scale_jitter(images, annotations, img_size=640, scale_range=(0.1, 2.0), aspect_ratio_range=(3./4, 4./3)):
    """
    大尺度抖动数据增强
    随机缩放和调整宽高比
    """
    augmented_images = []
    augmented_annotations = []
    
    for img, ann in zip(images, annotations):
        # 随机缩放因子
        scale = np.random.uniform(scale_range[0], scale_range[1])
        
        # 随机宽高比
        aspect_ratio = np.random.uniform(aspect_ratio_range[0], aspect_ratio_range[1])
        
        # 计算新的宽高
        new_h = int(img_size * scale)
        new_w = int(new_h * aspect_ratio)
        
        # 调整图像大小
        img = resize_image(img, (new_h, new_w))
        
        # 计算裁剪或填充区域
        if new_h > img_size or new_w > img_size:
            # 裁剪
            start_h = np.random.randint(0, new_h - img_size + 1)
            start_w = np.random.randint(0, new_w - img_size + 1)
            img = img[start_h:start_h + img_size, start_w:start_w + img_size]
            
            # 调整标注
            for bbox in ann:
                x, y, w, h, angle, class_id = bbox
                
                # 转换坐标
                x = x * new_w / img.shape[1] - start_w
                y = y * new_h / img.shape[0] - start_h
                w = w * new_w / img.shape[1]
                h = h * new_h / img.shape[0]
                
                # 检查边界框是否在裁剪后的图像内
                if x - w/2 >= 0 and x + w/2 <= img_size and y - h/2 >= 0 and y + h/2 <= img_size:
                    augmented_annotations.append([x, y, w, h, angle, class_id])
        else:
            # 填充
            pad_h = (img_size - new_h) // 2
            pad_w = (img_size - new_w) // 2
            padded_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            padded_img[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img
            
            # 调整标注
            for bbox in ann:
                x, y, w, h, angle, class_id = bbox
                
                # 转换坐标
                x = x * new_w / img.shape[1] + pad_w
                y = y * new_h / img.shape[0] + pad_h
                w = w * new_w / img.shape[1]
                h = h * new_h / img.shape[0]
                
                augmented_annotations.append([x, y, w, h, angle, class_id])
            
            img = padded_img
        
        augmented_images.append(img)
    
    return augmented_images, augmented_annotations

# 跨域学习辅助工具

def domain_adaptation_loss(source_features, target_features, lambda_=0.1):
    """
    计算域适应损失
    """
    # 计算源域和目标域特征的均值和方差
    source_mean = torch.mean(source_features, dim=0)
    target_mean = torch.mean(target_features, dim=0)
    source_var = torch.var(source_features, dim=0)
    target_var = torch.var(target_features, dim=0)
    
    # 计算均值差异损失和方差差异损失
    mean_loss = torch.mean(torch.abs(source_mean - target_mean))
    var_loss = torch.mean(torch.abs(torch.sqrt(source_var) - torch.sqrt(target_var)))
    
    # 总域适应损失
    da_loss = lambda_ * (mean_loss + var_loss)
    
    return da_loss

def adversarial_domain_loss(source_logits, target_logits):
    """
    计算对抗性域损失
    """
    # 源域和目标域预测概率
    source_prob = F.softmax(source_logits, dim=1)
    target_prob = F.softmax(target_logits, dim=1)
    
    # 计算域分类损失（最大化源域和目标域的预测差异）
    domain_loss = -torch.mean(torch.log(source_prob[:, 0] + 1e-8) + torch.log(1 - target_prob[:, 0] + 1e-8))
    
    return domain_loss

if __name__ == '__main__':
    # 测试部分工具函数
    
    # 测试角度转换
    angle = 45
    sin_val, cos_val = angle_to_sin_cos(angle)
    print(f"角度 {angle} 度对应的正弦值: {sin_val:.4f}, 余弦值: {cos_val:.4f}")
    
    # 测试旋转边界框IoU计算
    box1 = [320, 320, 100, 50, 0]
    box2 = [320, 320, 100, 50, 30]
    iou = rotated_iou(box1, box2)
    print(f"两个旋转边界框的IoU: {iou:.4f}")
    
    # 测试旋转非极大值抑制
    detections = [
        [320, 320, 100, 50, 0, 0.9, 0],
        [325, 325, 100, 50, 0, 0.8, 0],
        [500, 500, 80, 40, 15, 0.7, 1]
    ]
    nms_result = rotated_nms(detections, iou_threshold=0.5)
    print(f"NMS结果数量: {len(nms_result)}")