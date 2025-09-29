#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

from config import Config
from data_loader import SARImageDataset
from model import SAROrientedDetector
import utils

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='SAR图像多类别有向目标检测模型训练')
    parser.add_argument('--config', type=str, default='config.py', help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    parser.add_argument('--quick_demo', type=bool, default=False, help='快速演示模式')
    parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练的本地进程排名')
    return parser.parse_args()

# 设置随机种子
def set_seed(seed=42):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 加载数据
def get_data_loaders(config, distributed=False):
    """获取训练和验证数据加载器"""
    # 创建数据集
    train_dataset = SARImageDataset(
        root_dir=config.DATASET.ROOT_DIR,
        annotations_file=config.DATASET.TRAIN_ANNOTATIONS,
        image_dir=config.DATASET.IMAGE_DIR,
        classes=config.DATASET.CLASSES,
        image_size=config.DATASET.IMAGE_SIZE,
        is_train=True,
        augment=config.DATA_AUGMENTATION.ENABLE
    )
    
    val_dataset = SARImageDataset(
        root_dir=config.DATASET.ROOT_DIR,
        annotations_file=config.DATASET.VAL_ANNOTATIONS,
        image_dir=config.DATASET.IMAGE_DIR,
        classes=config.DATASET.CLASSES,
        image_size=config.DATASET.IMAGE_SIZE,
        is_train=False,
        augment=False
    )
    
    # 创建数据加载器
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DATASET.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=config.DATASET.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=utils.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DATASET.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATASET.NUM_WORKERS,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=utils.collate_fn
    )
    
    return train_loader, val_loader, train_sampler

# 初始化模型
def init_model(config, device):
    """初始化模型"""
    model = SAROrientedDetector(
        num_classes=config.DATASET.NUM_CLASSES,
        in_channels=config.MODEL.IN_CHANNELS,
        backbone_type=config.MODEL.BACKBONE,
        neck_type=config.MODEL.NECK,
        head_type=config.MODEL.HEAD
    )
    
    # 如果有预训练权重，加载预训练权重
    if config.MODEL.PRETRAINED and config.MODEL.PRETRAINED_PATH:
        try:
            pretrained_dict = torch.load(config.MODEL.PRETRAINED_PATH, map_location=device)
            model_dict = model.state_dict()
            # 过滤掉不匹配的层
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"成功加载预训练权重: {config.MODEL.PRETRAINED_PATH}")
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    
    # 移动模型到设备
    model.to(device)
    
    return model

# 初始化优化器
def init_optimizer(model, config):
    """初始化优化器"""
    # 获取需要优化的参数
    params = [p for p in model.parameters() if p.requires_grad]
    
    # 根据配置选择优化器
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            nesterov=config.TRAIN.NESTEROV
        )
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            params,
            lr=config.TRAIN.LR,
            betas=config.TRAIN.BETAS,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
    elif config.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=config.TRAIN.LR,
            betas=config.TRAIN.BETAS,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"不支持的优化器类型: {config.TRAIN.OPTIMIZER}")
    
    return optimizer

# 初始化学习率调度器
def init_lr_scheduler(optimizer, config, train_loader, start_epoch=0):
    """初始化学习率调度器"""
    if config.TRAIN.LR_SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.TRAIN.EPOCHS - start_epoch,
            eta_min=config.TRAIN.MIN_LR
        )
    elif config.TRAIN.LR_SCHEDULER == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.TRAIN.STEP_SIZE,
            gamma=config.TRAIN.GAMMA
        )
    elif config.TRAIN.LR_SCHEDULER == 'multi_step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.TRAIN.MILESTONES,
            gamma=config.TRAIN.GAMMA
        )
    elif config.TRAIN.LR_SCHEDULER == 'poly':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / config.TRAIN.EPOCHS) ** config.TRAIN.POLY_POWER
        )
    elif config.TRAIN.LR_SCHEDULER == 'warmup_cosine':
        # 预热学习率
        warmup_epochs = config.TRAIN.WARMUP_EPOCHS
        warmup_factor = config.TRAIN.WARMUP_FACTOR
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs
            else:
                # 余弦退火
                progress = (epoch - warmup_epochs) / (config.TRAIN.EPOCHS - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"不支持的学习率调度器类型: {config.TRAIN.LR_SCHEDULER}")
    
    return scheduler

# 计算损失
def compute_loss(outputs, targets, config):
    """计算模型损失"""
    # 分类损失
    cls_preds = outputs['cls_preds']
    cls_targets = targets['cls_targets']
    
    # 回归损失
    reg_preds = outputs['reg_preds']
    reg_targets = targets['reg_targets']
    
    # 角度损失
    angle_preds = outputs['angle_preds']
    angle_targets = targets['angle_targets']
    
    # 忽略背景类
    pos_mask = cls_targets > 0
    
    # 分类损失（Focal Loss）
    cls_loss_fn = utils.FocalLoss(
        alpha=config.LOSS.FOCAL_ALPHA,
        gamma=config.LOSS.FOCAL_GAMMA,
        reduction='mean'
    )
    cls_loss = cls_loss_fn(cls_preds, cls_targets)
    
    # 回归损失（GIoU Loss）
    if pos_mask.sum() > 0:
        reg_loss_fn = utils.GIoULoss(reduction='mean')
        reg_loss = reg_loss_fn(
            reg_preds[pos_mask], 
            reg_targets[pos_mask]
        )
    else:
        reg_loss = torch.tensor(0.0, device=cls_preds.device)
    
    # 角度损失（Circular Loss）
    if pos_mask.sum() > 0:
        angle_loss_fn = utils.CircularLoss(reduction='mean')
        angle_loss = angle_loss_fn(
            angle_preds[pos_mask], 
            angle_targets[pos_mask]
        )
    else:
        angle_loss = torch.tensor(0.0, device=cls_preds.device)
    
    # 总损失
    total_loss = (
        config.LOSS.CLS_WEIGHT * cls_loss +
        config.LOSS.REG_WEIGHT * reg_loss +
        config.LOSS.ANGLE_WEIGHT * angle_loss
    )
    
    return {
        'total_loss': total_loss,
        'cls_loss': cls_loss,
        'reg_loss': reg_loss,
        'angle_loss': angle_loss
    }

# 验证模型
def validate(model, val_loader, config, device, epoch=0):
    """验证模型性能"""
    model.eval()
    
    all_predictions = []
    all_annotations = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            # 移动数据到设备
            images = images.to(device)
            
            # 前向传播
            with autocast(enabled=config.TRAIN.MIXED_PRECISION):
                outputs = model(images)
            
            # 后处理
            detections = utils.postprocess_detections(
                outputs, 
                confidence_threshold=config.TEST.CONFIDENCE_THRESHOLD,
                iou_threshold=config.TEST.IOU_THRESHOLD,
                max_detections=config.TEST.MAX_DETECTIONS,
                use_rotated_nms=config.TEST.USE_ROTATED_NMS
            )
            
            # 收集预测结果和真实标注
            for i, dets in enumerate(detections):
                image_id = targets['image_ids'][i]
                height = targets['heights'][i]
                width = targets['widths'][i]
                
                # 转换预测结果到原始图像尺寸
                scaled_dets = utils.scale_detections(
                    dets, 
                    config.DATASET.IMAGE_SIZE, 
                    (height, width)
                )
                
                # 添加到预测结果列表
                for det in scaled_dets:
                    all_predictions.append({
                        'image_id': image_id,
                        'category_id': int(det[5]),
                        'bbox': det[:5].tolist(),  # [x, y, w, h, angle]
                        'score': float(det[4])
                    })
                
                # 添加真实标注
                gt_boxes = targets['bboxes'][i]
                gt_labels = targets['labels'][i]
                
                for box, label in zip(gt_boxes, gt_labels):
                    all_annotations.append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': box.tolist(),  # [x, y, w, h, angle]
                        'area': float(box[2] * box[3]),
                        'iscrowd': 0
                    })
    
    # 计算评估指标
    metrics = utils.compute_metrics(
        all_predictions, 
        all_annotations, 
        config.DATASET.CLASSES,
        iou_threshold=config.TEST.IOU_THRESHOLD,
        use_rotated_iou=config.TEST.USE_ROTATED_IOU
    )
    
    model.train()
    
    return metrics

# 训练一个epoch
def train_one_epoch(model, train_loader, optimizer, scheduler, config, device, epoch, scaler=None):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    cls_loss_sum = 0.0
    reg_loss_sum = 0.0
    angle_loss_sum = 0.0
    
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        # 移动数据到设备
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        with autocast(enabled=config.TRAIN.MIXED_PRECISION and scaler is not None):
            outputs = model(images)
            loss_dict = compute_loss(outputs, targets, config)
        
        # 反向传播和优化
        if config.TRAIN.MIXED_PRECISION and scaler is not None:
            scaler.scale(loss_dict['total_loss']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict['total_loss'].backward()
            optimizer.step()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 累计损失
        total_loss += loss_dict['total_loss'].item()
        cls_loss_sum += loss_dict['cls_loss'].item()
        reg_loss_sum += loss_dict['reg_loss'].item()
        angle_loss_sum += loss_dict['angle_loss'].item()
        
        # 打印训练进度
        if batch_idx % config.TRAIN.PRINT_FREQ == 0:
            current_lr = optimizer.param_groups[0]['lr']
            elapsed_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{config.TRAIN.EPOCHS}, Batch {batch_idx}/{len(train_loader)}, "
                  f"LR: {current_lr:.6f}, "
                  f"Total Loss: {total_loss/(batch_idx+1):.4f}, "
                  f"Cls Loss: {cls_loss_sum/(batch_idx+1):.4f}, "
                  f"Reg Loss: {reg_loss_sum/(batch_idx+1):.4f}, "
                  f"Angle Loss: {angle_loss_sum/(batch_idx+1):.4f}, "
                  f"Time: {elapsed_time:.2f}s")
    
    # 计算平均损失
    avg_total_loss = total_loss / len(train_loader)
    avg_cls_loss = cls_loss_sum / len(train_loader)
    avg_reg_loss = reg_loss_sum / len(train_loader)
    avg_angle_loss = angle_loss_sum / len(train_loader)
    
    return {
        'total_loss': avg_total_loss,
        'cls_loss': avg_cls_loss,
        'reg_loss': avg_reg_loss,
        'angle_loss': avg_angle_loss
    }

# 保存模型
def save_model(model, optimizer, scheduler, epoch, metrics, config, output_dir, is_best=False, distributed=False):
    """保存模型"""
    # 创建模型保存目录
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 准备保存的数据
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if distributed else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
        'config': config.__dict__
    }
    
    # 保存最新模型
    latest_path = os.path.join(models_dir, 'latest_model.pth')
    torch.save(save_dict, latest_path)
    
    # 保存当前epoch的模型
    epoch_path = os.path.join(models_dir, f'epoch_{epoch}_model.pth')
    torch.save(save_dict, epoch_path)
    
    # 如果是最佳模型，保存为最佳模型
    if is_best:
        best_path = os.path.join(models_dir, 'best_model.pth')
        torch.save(save_dict, best_path)
        print(f"保存最佳模型到 {best_path}")
    
    print(f"保存模型到 {epoch_path}")

# 保存训练日志
def save_logs(train_logs, val_logs, output_dir):
    """保存训练和验证日志"""
    # 创建日志目录
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # 保存日志为JSON文件
    logs_data = {
        'train_logs': train_logs,
        'val_logs': val_logs
    }
    
    logs_path = os.path.join(logs_dir, 'training_logs.json')
    with open(logs_path, 'w', encoding='utf-8') as f:
        json.dump(logs_data, f, ensure_ascii=False, indent=2)
    
    # 绘制损失和指标曲线
    plot_metrics(train_logs, val_logs, logs_dir)

# 绘制指标曲线
def plot_metrics(train_logs, val_logs, logs_dir):
    """绘制训练和验证指标曲线"""
    epochs = len(train_logs)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    
    # 总损失
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs+1), [log['total_loss'] for log in train_logs], 'b-', label='训练总损失')
    if val_logs:
        plt.plot(range(1, epochs+1), [log.get('total_loss', 0) for log in val_logs], 'r-', label='验证总损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.title('总损失曲线')
    plt.legend()
    
    # 分类损失
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs+1), [log['cls_loss'] for log in train_logs], 'b-', label='训练分类损失')
    if val_logs:
        plt.plot(range(1, epochs+1), [log.get('cls_loss', 0) for log in val_logs], 'r-', label='验证分类损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.title('分类损失曲线')
    plt.legend()
    
    # mAP
    plt.subplot(2, 2, 3)
    if val_logs and 'mAP' in val_logs[0]:
        plt.plot(range(1, epochs+1), [log['mAP'] for log in val_logs], 'r-', label='验证mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP曲线')
    plt.legend()
    
    # F1 Score
    plt.subplot(2, 2, 4)
    if val_logs and 'f1_score' in val_logs[0]:
        plt.plot(range(1, epochs+1), [log['f1_score'] for log in val_logs], 'r-', label='验证F1分数')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1分数曲线')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir, 'metrics_curves.png'))
    plt.close()

# 分布式训练入口
def distributed_main(local_rank, nprocs, args, config):
    """分布式训练入口函数"""
    # 初始化分布式环境
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23456',
        world_size=nprocs,
        rank=local_rank
    )
    
    # 设置CUDA设备
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    # 初始化模型
    model = init_model(config, device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 初始化优化器
    optimizer = init_optimizer(model, config)
    
    # 获取数据加载器
    train_loader, val_loader, train_sampler = get_data_loaders(config, distributed=True)
    
    # 初始化学习率调度器
    start_epoch = 0
    scheduler = init_lr_scheduler(optimizer, config, train_loader, start_epoch)
    
    # 恢复训练
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"成功恢复训练，从epoch {start_epoch}开始")
        else:
            print(f"警告: 未找到恢复文件 {args.resume}")
    
    # 初始化混合精度训练
    scaler = GradScaler() if config.TRAIN.MIXED_PRECISION else None
    
    # 训练循环
    best_map = 0.0
    train_logs = []
    val_logs = []
    
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        # 设置sampler的epoch
        train_sampler.set_epoch(epoch)
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, config, device, epoch, scaler
        )
        train_logs.append(train_metrics)
        
        # 验证模型
        if local_rank == 0 and (epoch + 1) % config.TRAIN.VAL_FREQ == 0:
            val_metrics = validate(model, val_loader, config, device, epoch)
            val_logs.append(val_metrics)
            
            # 打印验证结果
            print(f"Epoch {epoch+1}/{config.TRAIN.EPOCHS} 验证结果:")
            print(f"  mAP: {val_metrics.get('mAP', 0):.4f}")
            print(f"  Precision: {val_metrics.get('precision', 0):.4f}")
            print(f"  Recall: {val_metrics.get('recall', 0):.4f}")
            print(f"  F1 Score: {val_metrics.get('f1_score', 0):.4f}")
            
            # 保存模型
            current_map = val_metrics.get('mAP', 0)
            is_best = current_map > best_map
            if is_best:
                best_map = current_map
            
            save_model(model, optimizer, scheduler, epoch, val_metrics, config, args.output_dir, is_best, distributed=True)
            
            # 保存日志
            save_logs(train_logs, val_logs, args.output_dir)

# 主函数
def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed()
    
    # 加载配置
    config = Config()
    
    # 如果是快速演示模式，修改配置参数
    if args.quick_demo:
        print("启用快速演示模式，使用更轻量级的配置")
        config.TRAIN.EPOCHS = 5  # 更少的训练轮数
        config.DATASET.BATCH_SIZE = 2  # 更小的批次大小
        config.TRAIN.LR = 0.001  # 更大的学习率
        config.DATASET.NUM_WORKERS = 0  # 更少的工作进程
        config.TRAIN.PRINT_FREQ = 1  # 更频繁地打印进度
        config.TRAIN.VAL_FREQ = 1  # 更频繁地验证
        
        # 设置数据集路径为示例数据集
        config.DATASET.ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        config.DATASET.TRAIN_ANNOTATIONS = "annotations/train.json"
        config.DATASET.VAL_ANNOTATIONS = "annotations/val.json"
        config.DATASET.IMAGE_DIR = "images"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查是否使用分布式训练
    if config.TRAIN.DISTRIBUTED and torch.cuda.device_count() > 1:
        print(f"使用分布式训练，GPU数量: {torch.cuda.device_count()}")
        nprocs = torch.cuda.device_count()
        mp.spawn(
            distributed_main,
            nprocs=nprocs,
            args=(nprocs, args, config)
        )
    else:
        # 单卡训练
        # 初始化模型
        model = init_model(config, device)
        
        # 初始化优化器
        optimizer = init_optimizer(model, config)
        
        # 获取数据加载器
        train_loader, val_loader, _ = get_data_loaders(config, distributed=False)
        
        # 初始化学习率调度器
        start_epoch = 0
        scheduler = init_lr_scheduler(optimizer, config, train_loader, start_epoch)
        
        # 恢复训练
        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location=device)
                start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if checkpoint['scheduler_state_dict'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"成功恢复训练，从epoch {start_epoch}开始")
            else:
                print(f"警告: 未找到恢复文件 {args.resume}")
        
        # 初始化混合精度训练
        scaler = GradScaler() if config.TRAIN.MIXED_PRECISION else None
        
        # 训练循环
        best_map = 0.0
        train_logs = []
        val_logs = []
        
        for epoch in range(start_epoch, config.TRAIN.EPOCHS):
            # 训练一个epoch
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler, config, device, epoch, scaler
            )
            train_logs.append(train_metrics)
            
            # 验证模型
            if (epoch + 1) % config.TRAIN.VAL_FREQ == 0:
                val_metrics = validate(model, val_loader, config, device, epoch)
                val_logs.append(val_metrics)
                
                # 打印验证结果
                print(f"Epoch {epoch+1}/{config.TRAIN.EPOCHS} 验证结果:")
                print(f"  mAP: {val_metrics.get('mAP', 0):.4f}")
                print(f"  Precision: {val_metrics.get('precision', 0):.4f}")
                print(f"  Recall: {val_metrics.get('recall', 0):.4f}")
                print(f"  F1 Score: {val_metrics.get('f1_score', 0):.4f}")
                
                # 保存模型
                current_map = val_metrics.get('mAP', 0)
                is_best = current_map > best_map
                if is_best:
                    best_map = current_map
                
                save_model(model, optimizer, scheduler, epoch, val_metrics, config, args.output_dir, is_best)
                
                # 保存日志
                save_logs(train_logs, val_logs, args.output_dir)

if __name__ == '__main__':
    main()