# -*- coding: utf-8 -*-
"""
SAR图像多类别有向目标检测模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import math
import numpy as np

class CSPDarknetBlock(nn.Module):
    """
    CSPDarknet基本块，用于局部特征提取
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        
        # 主分支
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.relu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        
        # 捷径分支
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels // 2)
            )
        else:
            self.shortcut = nn.Identity()
        
        # 合并后的卷积
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # 主分支
        main = self.conv1(x)
        main = self.bn1(main)
        main = self.relu(main)
        
        main = self.conv2(main)
        main = self.bn2(main)
        main = self.relu(main)
        
        # 捷径分支
        shortcut = self.shortcut(x)
        
        # 合并特征
        merged = torch.cat([main, shortcut], dim=1)
        
        # 最终卷积
        out = self.conv3(merged)
        out = self.bn3(out)
        out = self.relu(out)
        
        return out

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer基本块，用于全局上下文建模
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        # 多头自注意力
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 窗口掩码（用于移位窗口注意力）
        self.register_buffer("attn_mask", None)
    
    def forward(self, x):
        # 输入形状: [B, C, H, W] -> 转换为 [B*num_windows, window_size*window_size, C]
        B, C, H, W = x.shape
        
        # 重塑为序列
        x_flatten = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # 自注意力
        attn_input = self.norm1(x_flatten)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, attn_mask=self.attn_mask)
        x = x_flatten + attn_output
        
        # 前馈网络
        ffn_input = self.norm2(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output
        
        # 重塑回原始形状
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return x

class HybridBackbone(nn.Module):
    """
    混合CNN-Transformer骨干网络
    结合CSPDarknet和Swin Transformer的优势
    """
    def __init__(self, in_channels=3, out_channels=[64, 128, 256, 512]):
        super().__init__()
        
        # 初始卷积
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU(0.1)
        
        # 阶段1: CSPDarknet块
        self.stage1 = self._make_stage(32, out_channels[0], num_blocks=3, stride=2)
        
        # 阶段2: CSPDarknet块
        self.stage2 = self._make_stage(out_channels[0], out_channels[1], num_blocks=6, stride=2)
        
        # 阶段3: 混合CSPDarknet和Swin Transformer块
        self.stage3 = nn.Sequential(
            CSPDarknetBlock(out_channels[1], out_channels[2], stride=2),
            SwinTransformerBlock(dim=out_channels[2], num_heads=4, window_size=7)
        )
        
        # 阶段4: Swin Transformer块
        self.stage4 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[3]),
            nn.LeakyReLU(0.1),
            SwinTransformerBlock(dim=out_channels[3], num_heads=8, window_size=7)
        )
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """创建CSPDarknet阶段"""
        layers = []
        # 降采样块
        layers.append(CSPDarknetBlock(in_channels, out_channels, stride=stride))
        # 添加多个CSPDarknet块
        for _ in range(num_blocks - 1):
            layers.append(CSPDarknetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 阶段1
        x1 = self.stage1(x)
        
        # 阶段2
        x2 = self.stage2(x1)
        
        # 阶段3
        x3 = self.stage3(x2)
        
        # 阶段4
        x4 = self.stage4(x3)
        
        # 返回多尺度特征
        return [x1, x2, x3, x4]

class FPN_PAN(nn.Module):
    """
    FPN-PAN颈部网络，用于多尺度特征融合
    """
    def __init__(self, in_channels=[64, 128, 256, 512], out_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 自上而下路径 (FPN)
        self.top_down_layers = nn.ModuleList()
        for i in range(len(in_channels)-1, 0, -1):
            self.top_down_layers.append(nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            ))
        
        # 自下而上路径 (PAN)
        self.bottom_up_layers = nn.ModuleList()
        for i in range(1, len(in_channels)):
            self.bottom_up_layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            ))
        
        # 输出卷积
        self.out_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.out_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            ))
    
    def forward(self, features):
        # 自上而下路径 (FPN)
        fpn_features = [None] * len(features)
        fpn_features[-1] = self.top_down_layers[0](features[-1])
        
        for i in range(len(features)-2, -1, -1):
            # 上采样
            upsampled = F.interpolate(fpn_features[i+1], scale_factor=2, mode='nearest')
            # 横向连接
            lateral = self.top_down_layers[len(features)-1 - i](features[i])
            # 特征融合
            fpn_features[i] = upsampled + lateral
        
        # 自下而上路径 (PAN)
        pan_features = [None] * len(features)
        pan_features[0] = fpn_features[0]
        
        for i in range(1, len(features)):
            # 下采样
            downsampled = F.max_pool2d(pan_features[i-1], kernel_size=2, stride=2)
            # 特征融合
            pan_features[i] = downsampled + fpn_features[i]
            # 卷积处理
            pan_features[i] = self.bottom_up_layers[i-1](pan_features[i])
        
        # 输出卷积
        out_features = []
        for i in range(len(features)):
            out_features.append(self.out_convs[i](pan_features[i]))
        
        return out_features

class ChannelAttention(nn.Module):
    """
    通道注意力模块
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    空间注意力模块
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    """
    卷积块注意力模块 (CBAM)
    结合通道注意力和空间注意力
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class OrientedDetectionHead(nn.Module):
    """
    有向目标检测头
    用于预测有向边界框、类别和角度
    """
    def __init__(self, in_channels, num_classes, feat_channels=256):
        super().__init__()
        self.num_classes = num_classes
        
        # 特征增强
        self.conv1 = nn.Conv2d(in_channels, feat_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feat_channels)
        self.relu = nn.LeakyReLU(0.1)
        
        # CBAM注意力模块
        self.cbam = CBAM(feat_channels)
        
        # 分类分支
        self.cls_conv = nn.Conv2d(feat_channels, num_classes, kernel_size=3, stride=1, padding=1)
        
        # 边界框回归分支（4个坐标）
        self.reg_conv = nn.Conv2d(feat_channels, 4, kernel_size=3, stride=1, padding=1)
        
        # 角度回归分支（使用正弦和余弦表示，避免边界不连续）
        self.angle_conv = nn.Conv2d(feat_channels, 2, kernel_size=3, stride=1, padding=1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征增强
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 注意力增强
        x = self.cbam(x)
        
        # 分类预测
        cls_logits = self.cls_conv(x)
        
        # 边界框回归预测
        bbox_reg = self.reg_conv(x)
        
        # 角度预测（正弦和余弦表示）
        angle_reg = self.angle_conv(x)
        
        return cls_logits, bbox_reg, angle_reg

class SAROrientedDetector(nn.Module):
    """
    SAR图像多类别有向目标检测模型
    集成骨干网络、颈部网络和检测头
    """
    def __init__(self, num_classes=5, in_channels=3):
        super().__init__()
        
        # 骨干网络
        self.backbone = HybridBackbone(in_channels=in_channels)
        
        # 颈部网络
        self.neck = FPN_PAN()
        
        # 检测头（针对不同尺度的特征图）
        self.heads = nn.ModuleList()
        for _ in range(4):  # 4个尺度的特征图
            self.heads.append(OrientedDetectionHead(in_channels=256, num_classes=num_classes))
    
    def forward(self, x):
        # 骨干网络提取特征
        features = self.backbone(x)
        
        # 颈部网络融合特征
        fused_features = self.neck(features)
        
        # 检测头预测
        cls_logits_list = []
        bbox_reg_list = []
        angle_reg_list = []
        
        for i, feat in enumerate(fused_features):
            cls_logits, bbox_reg, angle_reg = self.heads[i](feat)
            cls_logits_list.append(cls_logits)
            bbox_reg_list.append(bbox_reg)
            angle_reg_list.append(angle_reg)
        
        # 返回多尺度预测结果
        return cls_logits_list, bbox_reg_list, angle_reg_list

class FocalLoss(nn.Module):
    """
    Focal Loss实现，用于解决类别不平衡问题
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算焦点损失权重
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 损失聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CircularLoss(nn.Module):
    """
    圆形损失，用于处理角度回归的周期性
    使用正弦和余弦表示角度，避免边界不连续问题
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        # 确保预测和目标形状一致
        assert pred.shape == target.shape
        
        # 计算角度差的正弦和余弦
        sin_diff = pred[:, 0] - target[:, 0]
        cos_diff = pred[:, 1] - target[:, 1]
        
        # 计算损失（L2损失）
        loss = torch.sqrt(sin_diff ** 2 + cos_diff ** 2)
        
        # 损失聚合
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class GIoULoss(nn.Module):
    """
    GIoU损失实现
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes):
        # 计算IoU
        iou = self._calculate_iou(pred_boxes, target_boxes)
        
        # 计算GIoU
        giou = self._calculate_giou(pred_boxes, target_boxes, iou)
        
        # GIoU损失 = 1 - GIoU
        loss = 1 - giou
        
        # 损失聚合
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _calculate_iou(self, boxes1, boxes2):
        # 计算交集面积
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 0] + boxes1[:, 2], boxes2[:, 0] + boxes2[:, 2])
        y2 = torch.min(boxes1[:, 1] + boxes1[:, 3], boxes2[:, 1] + boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # 计算并集面积
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        union = area1 + area2 - intersection
        
        # 计算IoU
        iou = intersection / (union + 1e-8)
        
        return iou
    
    def _calculate_giou(self, boxes1, boxes2, iou):
        # 计算最小包围框
        x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.max(boxes1[:, 0] + boxes1[:, 2], boxes2[:, 0] + boxes2[:, 2])
        y2 = torch.max(boxes1[:, 1] + boxes1[:, 3], boxes2[:, 1] + boxes2[:, 3])
        
        # 计算最小包围框面积
        enclosing_area = (x2 - x1) * (y2 - y1)
        
        # 计算GIoU
        giou = iou - (enclosing_area - (boxes1[:, 2] * boxes1[:, 3] + boxes2[:, 2] * boxes2[:, 3] - (iou * (boxes1[:, 2] * boxes1[:, 3] + boxes2[:, 2] * boxes2[:, 3] - intersection)))) / (enclosing_area + 1e-8)
        
        return giou

class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    结合分类损失、回归损失、角度损失和IoU损失
    """
    def __init__(self, num_classes=5, class_weight=1.0, reg_weight=5.0, angle_weight=1.0, iou_weight=2.0):
        super().__init__()
        self.class_weight = class_weight
        self.reg_weight = reg_weight
        self.angle_weight = angle_weight
        self.iou_weight = iou_weight
        
        # 初始化各个损失函数
        self.cls_loss = FocalLoss()
        self.reg_loss = nn.SmoothL1Loss()
        self.angle_loss = CircularLoss()
        self.iou_loss = GIoULoss()
    
    def forward(self, outputs, targets):
        cls_logits_list, bbox_reg_list, angle_reg_list = outputs
        
        total_loss = 0.0
        
        # 处理每个尺度的预测结果
        for i in range(len(cls_logits_list)):
            cls_logits = cls_logits_list[i]
            bbox_reg = bbox_reg_list[i]
            angle_reg = angle_reg_list[i]
            
            # 这里简化处理，实际应用中需要根据正样本和负样本计算损失
            # ...
            
            # 计算分类损失
            # cls_loss = self.cls_loss(cls_logits, target_labels)
            
            # 计算回归损失
            # reg_loss = self.reg_loss(bbox_reg, target_boxes)
            
            # 计算角度损失
            # angle_loss = self.angle_loss(angle_reg, target_angles)
            
            # 计算IoU损失
            # iou_loss = self.iou_loss(pred_boxes, target_boxes)
            
            # 总损失
            # total_loss += self.class_weight * cls_loss + \
            #              self.reg_weight * reg_loss + \
            #              self.angle_weight * angle_loss + \
            #              self.iou_weight * iou_loss
        
        # 这里返回一个示例损失值，实际应用中需要替换为真实的损失计算
        return torch.tensor(0.5, requires_grad=True, device=cls_logits_list[0].device)

# 辅助函数：角度转换

def angle_to_sin_cos(angle):
    """将角度转换为正弦和余弦表示"""
    rad = math.radians(angle)
    return math.sin(rad), math.cos(rad)

def sin_cos_to_angle(sin_val, cos_val):
    """将正弦和余弦表示转换为角度"""
    rad = math.atan2(sin_val, cos_val)
    return math.degrees(rad)

# 辅助函数：旋转非极大值抑制

def rotated_nms(detections, iou_threshold=0.5):
    """
    旋转非极大值抑制
    过滤重叠的检测结果
    """
    # 按置信度排序
    if len(detections) == 0:
        return []
    
    # 简化实现，实际应用中需要根据旋转边界框计算IoU
    # ...
    
    return detections[:500]  # 返回前500个检测结果

if __name__ == '__main__':
    # 测试模型
    model = SAROrientedDetector(num_classes=5)
    
    # 创建一个模拟输入
    input_tensor = torch.randn(2, 3, 640, 640)
    
    # 前向传播
    cls_logits, bbox_reg, angle_reg = model(input_tensor)
    
    # 打印输出形状
    print(f"分类输出数量: {len(cls_logits)}")
    for i, logits in enumerate(cls_logits):
        print(f"  尺度 {i+1} 形状: {logits.shape}")
    
    print(f"边界框输出数量: {len(bbox_reg)}")
    for i, reg in enumerate(bbox_reg):
        print(f"  尺度 {i+1} 形状: {reg.shape}")
    
    print(f"角度输出数量: {len(angle_reg)}")
    for i, angle in enumerate(angle_reg):
        print(f"  尺度 {i+1} 形状: {angle.shape}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params/1e6:.2f}M")