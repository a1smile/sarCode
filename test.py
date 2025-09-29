# -*- coding: utf-8 -*-
"""
SAR图像多类别有向目标检测模型测试脚本
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入自定义模块
from config import Config
from data_loader import SARImageDataset
from model import SAROrientedDetector
import utils

# 设置随机种子
def set_seed(seed=42):
    """设置随机种子以保证实验可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 初始化日志
def init_logger(log_dir, log_name='test.log'):
    """初始化日志记录器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 文件输出
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 解码预测结果
def decode_predictions(outputs, config, device):
    """
    解码模型输出，得到最终的检测结果
    """
    cls_logits_list, bbox_reg_list, angle_reg_list = outputs
    
    detections = []
    
    # 遍历不同尺度的特征图
    for i in range(len(cls_logits_list)):
        cls_logits = cls_logits_list[i]
        bbox_reg = bbox_reg_list[i]
        angle_reg = angle_reg_list[i]
        
        # 获取特征图尺寸
        batch_size, num_classes, feat_h, feat_w = cls_logits.shape
        
        # 生成网格坐标
        yv, xv = torch.meshgrid(torch.arange(feat_h), torch.arange(feat_w))
        grid = torch.stack((xv, yv), 2).float().to(device)
        
        # 计算步长
        stride_h = config.DATASET.IMAGE_SIZE[0] / feat_h
        stride_w = config.DATASET.IMAGE_SIZE[1] / feat_w
        
        # 计算中心点坐标
        cx = (grid[..., 0] + 0.5) * stride_w
        cy = (grid[..., 1] + 0.5) * stride_h
        
        # 遍历每个批次
        for b in range(batch_size):
            # 遍历每个类别
            for c in range(num_classes):
                # 获取该类别的置信度
                conf = torch.sigmoid(cls_logits[b, c])
                
                # 过滤低置信度的预测
                if conf < config.TEST.CONFIDENCE_THRESHOLD:
                    continue
                
                # 获取边界框回归参数
                bx = bbox_reg[b, 0]
                by = bbox_reg[b, 1]
                bw = bbox_reg[b, 2]
                bh = bbox_reg[b, 3]
                
                # 获取角度回归参数
                sin_angle = angle_reg[b, 0]
                cos_angle = angle_reg[b, 1]
                
                # 计算角度（弧度转角度）
                angle = torch.atan2(sin_angle, cos_angle) * 180 / np.pi
                
                # 计算最终的边界框坐标
                x = cx + bx * stride_w
                y = cy + by * stride_h
                w = torch.exp(bw) * stride_w
                h = torch.exp(bh) * stride_h
                
                # 添加到检测结果中
                detections.append([
                    x.item(), y.item(), w.item(), h.item(), angle.item(),
                    conf.item(), c
                ])
    
    # 应用旋转非极大值抑制
    if len(detections) > 0:
        detections = utils.rotated_nms(
            detections, 
            iou_threshold=config.TEST.IOU_THRESHOLD
        )
    
    return detections

# 单张图像推理
def inference_single_image(model, image, config, device):
    """
    对单张图像进行推理
    """
    # 预处理图像
    img_h, img_w = image.shape[:2]
    
    # 调整图像大小
    resized_image = cv2.resize(image, (config.DATASET.IMAGE_SIZE[1], config.DATASET.IMAGE_SIZE[0]))
    
    # 转换为Tensor
    image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        # 混合精度推理
        with torch.cuda.amp.autocast(enabled=config.TEST.MIXED_PRECISION):
            outputs = model(image_tensor)
    
    # 解码预测结果
    detections = decode_predictions(outputs, config, device)
    
    # 调整坐标到原始图像尺寸
    scale_h = img_h / config.DATASET.IMAGE_SIZE[0]
    scale_w = img_w / config.DATASET.IMAGE_SIZE[1]
    
    for det in detections:
        det[0] *= scale_w  # x
        det[1] *= scale_h  # y
        det[2] *= scale_w  # w
        det[3] *= scale_h  # h
    
    return detections

# 测试模型
def test_model(model, dataloader, config, logger, device, result_dir):
    """
    在测试数据集上评估模型性能
    """
    model.eval()
    
    # 初始化计时器
    start_time = time.time()
    
    # 初始化评估指标
    all_predictions = []
    all_ground_truths = []
    
    # 进度条
    pbar = tqdm(dataloader, desc='测试', ncols=100)
    
    for batch_idx, (images, targets, image_paths) in enumerate(pbar):
        # 移动数据到设备
        images = images.to(device)
        
        # 推理
        with torch.no_grad():
            # 混合精度推理
            with torch.cuda.amp.autocast(enabled=config.TEST.MIXED_PRECISION):
                outputs = model(images)
        
        # 遍历批次中的每张图像
        for i in range(len(images)):
            # 获取原始图像路径
            image_path = image_paths[i]
            
            # 读取原始图像
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # 获取当前图像的输出
            batch_outputs = [output[i:i+1] for output in outputs]
            
            # 解码预测结果
            detections = decode_predictions(batch_outputs, config, device)
            
            # 调整坐标到原始图像尺寸
            img_h, img_w = original_image.shape[:2]
            scale_h = img_h / config.DATASET.IMAGE_SIZE[0]
            scale_w = img_w / config.DATASET.IMAGE_SIZE[1]
            
            scaled_detections = []
            for det in detections:
                scaled_det = det.copy()
                scaled_det[0] *= scale_w  # x
                scaled_det[1] *= scale_h  # y
                scaled_det[2] *= scale_w  # w
                scaled_det[3] *= scale_h  # h
                scaled_detections.append(scaled_det)
            
            # 添加到总预测结果中
            all_predictions.extend(scaled_detections)
            
            # 获取当前图像的真实标签
            # 这里需要根据实际的数据加载器返回格式进行调整
            # ground_truths = get_ground_truths(targets[i])
            # all_ground_truths.extend(ground_truths)
            
            # 可视化并保存结果
            if config.TEST.SAVE_VISUALIZATION:
                # 可视化检测结果
                visualized_image = utils.visualize_detections(
                    original_image, 
                    scaled_detections, 
                    config.DATASET.CLASSES, 
                    score_threshold=config.TEST.CONFIDENCE_THRESHOLD
                )
                
                # 保存可视化结果
                image_name = os.path.basename(image_path)
                save_path = os.path.join(result_dir, 'visualizations', image_name)
                cv2.imwrite(save_path, cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
            
            # 保存检测结果到文件
            if config.TEST.SAVE_RESULTS:
                result_file = os.path.join(result_dir, 'detections', os.path.splitext(image_name)[0] + '.txt')
                with open(result_file, 'w') as f:
                    for det in scaled_detections:
                        x, y, w, h, angle, score, class_id = det
                        class_name = config.DATASET.CLASSES[int(class_id)]
                        f.write(f'{class_name} {score:.4f} {x:.2f} {y:.2f} {w:.2f} {h:.2f} {angle:.2f}\n')
    
    # 计算推理时间
    elapsed_time = time.time() - start_time
    fps = len(dataloader.dataset) / elapsed_time
    
    # 计算评估指标
    # 这里需要根据实际的预测结果和真实标签格式进行调整
    metrics = {
        'mAP': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'fps': fps
    }
    
    # 记录日志
    logger.info(f'测试完成! 处理了 {len(dataloader.dataset)} 张图像，耗时 {elapsed_time:.2f} 秒，FPS: {fps:.2f}')
    logger.info(f'mAP: {metrics["mAP"]:.4f}, 精确率: {metrics["precision"]:.4f}, 召回率: {metrics["recall"]:.4f}, F1分数: {metrics["f1_score"]:.4f}')
    
    return metrics

# 主测试函数
def main():
    """主测试函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SAR图像多类别有向目标检测模型测试')
    parser.add_argument('--config', type=str, default='config.py', help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--test_data', type=str, default=None, help='测试数据集路径，覆盖配置文件中的设置')
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    if args.config and os.path.exists(args.config):
        config.load_from_file(args.config)
    
    # 如果指定了测试数据路径，则覆盖配置文件中的设置
    if args.test_data:
        config.DATASET.ROOT = args.test_data
    
    # 设置随机种子
    set_seed(config.SEED)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化日志
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = init_logger(log_dir, 'test.log')
    logger.info(f'配置: {config}')
    logger.info(f'使用设备: {device}')
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 创建结果保存目录
    result_dir = os.path.join(args.output_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 创建可视化结果保存目录
    if config.TEST.SAVE_VISUALIZATION:
        vis_dir = os.path.join(result_dir, 'visualizations')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
    
    # 创建检测结果保存目录
    if config.TEST.SAVE_RESULTS:
        det_dir = os.path.join(result_dir, 'detections')
        if not os.path.exists(det_dir):
            os.makedirs(det_dir)
    
    # 初始化模型
    model = SAROrientedDetector(
        num_classes=config.DATASET.NUM_CLASSES,
        in_channels=config.MODEL.IN_CHANNELS
    )
    
    # 加载模型权重
    logger.info(f'加载模型权重: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 处理DDP模型的键名
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        # 如果是DDP模型，移除'module.'前缀
        if k.startswith('module.'):
            new_k = k[7:]
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    
    # 加载权重
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # 加载测试数据
    test_dataset = SARImageDataset(
        root_dir=config.DATASET.ROOT,
        split='test',
        image_size=config.DATASET.IMAGE_SIZE,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.DATASET.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATASET.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f'测试数据集大小: {len(test_dataset)}')
    
    # 测试模型
    metrics = test_model(
        model=model,
        dataloader=test_loader,
        config=config,
        logger=logger,
        device=device,
        result_dir=result_dir
    )
    
    # 保存评估结果
    metrics_path = os.path.join(result_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'mAP: {metrics["mAP"]:.4f}\n')
        f.write(f'精确率: {metrics["precision"]:.4f}\n')
        f.write(f'召回率: {metrics["recall"]:.4f}\n')
        f.write(f'F1分数: {metrics["f1_score"]:.4f}\n')
        f.write(f'FPS: {metrics["fps"]:.2f}\n')
    
    logger.info(f'评估结果已保存到: {metrics_path}')

# 执行主函数
if __name__ == '__main__':
    main()