# -*- coding: utf-8 -*-
"""
SAR图像数据加载与预处理模块
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import random
from skimage import filters, exposure, morphology
import json

class SARImageDataset(Dataset):
    """
    SAR图像数据集类
    负责加载和预处理SAR图像数据
    """
    def __init__(self, root_dir, split='train', img_size=640, augmentation=True):
        """
        初始化数据集
        
        参数:
            root_dir: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            img_size: 输入图像尺寸
            augmentation: 是否启用数据增强
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augmentation = augmentation
        
        # 加载图像路径和标注
        self.img_paths, self.annotations = self._load_data()
        
        # 数据预处理和增强变换
        self.transform = self._get_transforms()
        
    def _load_data(self):
        """加载图像路径和标注"""
        img_paths = []
        annotations = []
        
        # 假设数据集组织结构为：
        # root_dir/
        #   images/
        #     train/
        #     val/
        #     test/
        #   annotations/
        #     train.json
        #     val.json
        #     test.json
        
        img_dir = os.path.join(self.root_dir, 'images', self.split)
        ann_file = os.path.join(self.root_dir, 'annotations', f'{self.split}.json')
        
        # 检查文件和目录是否存在
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
        
        if os.path.exists(ann_file):
            # 加载JSON格式标注
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 解析JSON格式
            img_id_to_info = {img['id']: img for img in data['images']}
            
            # 按图像分组标注
            anns_by_img_id = {}
            for ann in data['annotations']:
                img_id = ann['image_id']
                if img_id not in anns_by_img_id:
                    anns_by_img_id[img_id] = []
                anns_by_img_id[img_id].append(ann)
            
            # 构建图像路径和标注列表
            for img_id, img_info in img_id_to_info.items():
                img_path = os.path.join(img_dir, img_info['file_name'])
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                    if img_id in anns_by_img_id:
                        annotations.append(anns_by_img_id[img_id])
                    else:
                        annotations.append([])
        else:
            # 如果没有标注文件，只加载图像
            for img_name in os.listdir(img_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    img_path = os.path.join(img_dir, img_name)
                    img_paths.append(img_path)
                    annotations.append([])  # 无标注
        
        # 数据清洗
        img_paths, annotations = self._clean_data(img_paths, annotations)
        
        return img_paths, annotations
    
    def _clean_data(self, img_paths, annotations):
        """数据清洗
        1. 移除低质量图像
        2. 过滤标注异常值
        3. 去重
        """
        cleaned_img_paths = []
        cleaned_annotations = []
        
        # 记录已处理的图像路径，用于去重
        seen_paths = set()
        
        for img_path, anns in zip(img_paths, annotations):
            # 检查是否已处理过该图像（去重）
            if img_path in seen_paths:
                continue
            seen_paths.add(img_path)
            
            try:
                # 加载图像并检查质量
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.size == 0:
                    continue
                
                # 检查图像是否有过多的斑点噪声或几何畸变
                # 计算图像对比度
                min_val, max_val = img.min(), img.max()
                contrast = max_val - min_val
                if contrast < 10:  # 过低对比度
                    continue
                
                # 过滤标注异常值
                filtered_anns = []
                for ann in anns:
                    # 检查边界框尺寸是否合理
                    if 'bbox' in ann:
                        x, y, w, h = ann['bbox']
                        # 过滤过小或过大的边界框
                        if w < 5 or h < 5 or w > img.shape[1] * 0.5 or h > img.shape[0] * 0.5:
                            continue
                    
                    # 检查角度标注是否合理（如果有）
                    if 'angle' in ann and (ann['angle'] < -180 or ann['angle'] > 180):
                        continue
                    
                    filtered_anns.append(ann)
                
                cleaned_img_paths.append(img_path)
                cleaned_annotations.append(filtered_anns)
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                continue
        
        return cleaned_img_paths, cleaned_annotations
    
    def _get_transforms(self):
        """获取数据预处理和增强变换"""
        transform_list = []
        
        # 基础预处理
        transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.Resize((self.img_size, self.img_size)))
        
        if self.augmentation and self.split == 'train':
            # 训练集数据增强
            transform_list.extend([
                # 几何变换
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=360,  # 随机旋转0-360度
                        scale=(0.5, 1.5),  # 随机缩放
                        shear=(-10, 10)  # 随机剪切
                    )
                ], p=0.5),
                
                # 随机翻转
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                
                # 对比度调整
                transforms.RandomApply([
                    transforms.ColorJitter(contrast=(0.5, 1.5))
                ], p=0.5)
            ])
        
        # 转换为张量并归一化
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]
        ])
        
        return transforms.Compose(transform_list)
    
    def _intensity_normalization(self, img):
        """强度归一化"""
        # 最小-最大归一化
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    
    def _adaptive_histogram_equalization(self, img):
        """自适应直方图均衡化"""
        # 使用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply((img * 255).astype(np.uint8))
        return img_eq / 255.0
    
    def _speckle_noise_augmentation(self, img):
        """斑点噪声增强"""
        # 模拟SAR图像斑点噪声
        sigma = random.uniform(0.01, 0.1)
        noise = np.random.randn(*img.shape) * sigma
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 1)
        return noisy_img
    
    def _large_scale_jitter(self, img, bboxes=None):
        """大尺度抖动增强"""
        # 随机缩放比例
        scale = random.uniform(0.8, 1.2)
        
        # 计算新尺寸
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 缩放图像
        img = cv2.resize(img, (new_w, new_h))
        
        # 随机裁剪回原尺寸
        if new_h > h and new_w > w:
            y1 = random.randint(0, new_h - h)
            x1 = random.randint(0, new_w - w)
            img = img[y1:y1+h, x1:x1+w]
            
            # 调整边界框坐标（如果有）
            if bboxes is not None:
                new_bboxes = []
                for bbox in bboxes:
                    x, y, w_bbox, h_bbox, angle = bbox
                    new_x = x * scale - x1
                    new_y = y * scale - y1
                    new_w_bbox = w_bbox * scale
                    new_h_bbox = h_bbox * scale
                    new_bboxes.append([new_x, new_y, new_w_bbox, new_h_bbox, angle])
                return img, new_bboxes
        
        return img, bboxes
    
    def _load_and_preprocess_image(self, img_path):
        """加载并预处理图像"""
        # 加载图像（灰度图）
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {img_path}")
        
        # 强度归一化
        img = self._intensity_normalization(img)
        
        # 自适应直方图均衡化
        img = self._adaptive_histogram_equalization(img)
        
        # 转换为RGB格式（3通道）以适应模型输入
        img = np.stack([img] * 3, axis=-1)
        
        return img
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.img_paths[idx]
        anns = self.annotations[idx].copy()
        
        # 加载并预处理图像
        img = self._load_and_preprocess_image(img_path)
        
        # 提取标注信息
        targets = {
            'image_id': idx,
            'labels': [],
            'boxes': [],
            'angles': []
        }
        
        for ann in anns:
            # 类别标签
            if 'category_id' in ann:
                targets['labels'].append(ann['category_id'])
            
            # 边界框坐标
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                targets['boxes'].append([x, y, w, h])
            
            # 旋转角度（如果有）
            if 'angle' in ann:
                targets['angles'].append(ann['angle'])
            else:
                # 如果没有角度信息，默认为0
                targets['angles'].append(0.0)
        
        # 数据增强
        if self.augmentation and self.split == 'train':
            # 转换为PIL图像以应用torchvision变换
            img_pil = transforms.ToPILImage()(img)
            img = self.transform(img_pil)
            
            # 斑点噪声增强
            if random.random() < 0.3:
                img_np = img.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                img_np = self._speckle_noise_augmentation(img_np)
                img = torch.from_numpy(img_np.transpose(2, 0, 1))  # (H, W, C) -> (C, H, W)
            
            # 大尺度抖动增强
            if random.random() < 0.3 and targets['boxes']:
                img_np = img.numpy().transpose(1, 2, 0)
                # 这里简化处理，实际应用中需要根据增强调整边界框
                img_np, _ = self._large_scale_jitter(img_np)
                img = torch.from_numpy(img_np.transpose(2, 0, 1))
        else:
            # 测试集或验证集仅进行基本预处理
            img_pil = transforms.ToPILImage()(img)
            img = self.transform(img_pil)
        
        # 转换目标为张量
        if targets['labels']:
            targets['labels'] = torch.tensor(targets['labels'], dtype=torch.long)
            targets['boxes'] = torch.tensor(targets['boxes'], dtype=torch.float32)
            targets['angles'] = torch.tensor(targets['angles'], dtype=torch.float32)
        else:
            # 如果没有目标，创建空张量
            targets['labels'] = torch.tensor([], dtype=torch.long)
            targets['boxes'] = torch.tensor([], dtype=torch.float32).reshape(0, 4)
            targets['angles'] = torch.tensor([], dtype=torch.float32)
        
        return img, targets, img_path

def get_data_loaders(cfg):
    """
    获取数据加载器
    
    参数:
        cfg: 配置对象
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 创建数据集
    train_dataset = SARImageDataset(
        root_dir=cfg.DATASET.ROOT,
        split=cfg.DATASET.TRAIN_SPLIT,
        img_size=cfg.DATASET.IMG_SIZE,
        augmentation=cfg.AUGMENTATION.ENABLE
    )
    
    val_dataset = SARImageDataset(
        root_dir=cfg.DATASET.ROOT,
        split=cfg.DATASET.VAL_SPLIT,
        img_size=cfg.DATASET.IMG_SIZE,
        augmentation=False
    )
    
    test_dataset = SARImageDataset(
        root_dir=cfg.DATASET.ROOT,
        split=cfg.DATASET.TEST_SPLIT,
        img_size=cfg.DATASET.IMG_SIZE,
        augmentation=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATASET.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """
    自定义批处理函数，处理不同样本中目标数量不同的情况
    """
    images = []
    targets = []
    img_paths = []
    
    for img, target, img_path in batch:
        images.append(img)
        targets.append(target)
        img_paths.append(img_path)
    
    # 将图像堆叠成张量
    images = torch.stack(images, dim=0)
    
    return images, targets, img_paths

def compute_class_weights(dataset, num_classes):
    """
    计算类别权重，用于处理类别不平衡问题
    """
    class_counts = np.zeros(num_classes)
    
    for anns in dataset.annotations:
        for ann in anns:
            if 'category_id' in ann:
                cat_id = ann['category_id']
                if 0 <= cat_id < num_classes:
                    class_counts[cat_id] += 1
    
    # 计算权重（反比于频率）
    total_count = np.sum(class_counts)
    class_weights = total_count / (num_classes * (class_counts + 1e-8))
    
    # 归一化权重
    class_weights = class_weights / np.sum(class_weights)
    
    return torch.tensor(class_weights, dtype=torch.float32)

if __name__ == '__main__':
    # 测试数据加载器
    from config import parse_args
    
    cfg = parse_args()
    train_loader, val_loader, test_loader = get_data_loaders(cfg)
    
    # 打印数据集大小
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 测试加载一个批次
    for images, targets, img_paths in train_loader:
        print(f"批次图像形状: {images.shape}")
        print(f"批次目标数量: {len(targets)}")
        print(f"第一个样本的目标数量: {len(targets[0]['labels'])}")
        break