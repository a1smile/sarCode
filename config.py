# -*- coding: utf-8 -*-
"""
大规模SAR图像多类别有向目标检测配置文件
"""

import os
from yacs.config import CfgNode as CN

# 创建配置节点
_C = CN()

# 数据集配置
_C.DATASET = CN()
_C.DATASET.NAME = 'SARDet-100K'  # 数据集名称
_C.DATASET.ROOT = './data/SARDet-100K'  # 数据集根目录
_C.DATASET.CLASSES = ['ship', 'airplane', 'vehicle', 'building', 'other']  # 目标类别
_C.DATASET.NUM_CLASSES = len(_C.DATASET.CLASSES)  # 类别数量
_C.DATASET.IMG_SIZE = 640  # 输入图像尺寸
_C.DATASET.BATCH_SIZE = 16  # 批次大小
_C.DATASET.NUM_WORKERS = 4  # 数据加载线程数
_C.DATASET.TRAIN_SPLIT = 'train'  # 训练集划分
_C.DATASET.VAL_SPLIT = 'val'  # 验证集划分
_C.DATASET.TEST_SPLIT = 'test'  # 测试集划分

# 模型配置
_C.MODEL = CN()
_C.MODEL.TYPE = 'HybridCNNTransformer'  # 模型类型
_C.MODEL.BACKBONE = 'CSPDarknetSwin'  # 骨干网络
_C.MODEL.NECK = 'FPN-PAN'  # 颈部网络
_C.MODEL.HEAD = 'OrientedDetHead'  # 检测头
_C.MODEL.PRETRAINED = False  # 是否使用预训练权重
_C.MODEL.PRETRAINED_PATH = ''  # 预训练权重路径
_C.MODEL.ANCHOR_FREE = True  # 是否使用无锚点检测
_C.MODEL.NUM_FEATURE_LEVELS = 3  # 特征层级数量

# 训练配置
_C.TRAIN = CN()
_C.TRAIN.RESUME = False  # 是否从断点恢复训练
_C.TRAIN.RESUME_PATH = ''  # 断点路径
_C.TRAIN.EPOCHS = 300  # 训练轮数
_C.TRAIN.LR_INIT = 0.001  # 初始学习率
_C.TRAIN.LR_MIN = 0.00001  # 最小学习率
_C.TRAIN.WARMUP_EPOCHS = 5  # 预热轮数
_C.TRAIN.WARMUP_LR = 0.0001  # 预热学习率
_C.TRAIN.OPTIMIZER = 'AdamW'  # 优化器类型
_C.TRAIN.WEIGHT_DECAY = 0.0005  # 权重衰减
_C.TRAIN.MOMENTUM = 0.9  # 动量
_C.TRAIN.COSINE_ANNEALING = True  # 是否使用余弦退火学习率

# 损失函数配置
_C.LOSS = CN()
_C.LOSS.CLASS_LOSS = 'FocalLoss'  # 分类损失
_C.LOSS.REG_LOSS = 'SmoothL1'  # 回归损失
_C.LOSS.ANGLE_LOSS = 'CircularLoss'  # 角度损失
_C.LOSS.IOU_LOSS = 'GIoU'  # IoU损失
_C.LOSS.CLASS_WEIGHT = 1.0  # 分类损失权重
_C.LOSS.REG_WEIGHT = 5.0  # 回归损失权重
_C.LOSS.ANGLE_WEIGHT = 1.0  # 角度损失权重
_C.LOSS.IOU_WEIGHT = 2.0  # IoU损失权重

# 数据增强配置
_C.AUGMENTATION = CN()
_C.AUGMENTATION.ENABLE = True  # 是否启用数据增强
_C.AUGMENTATION.RANDOM_ROTATION = True  # 随机旋转
_C.AUGMENTATION.ROTATION_RANGE = 360  # 旋转范围
_C.AUGMENTATION.RANDOM_SCALE = True  # 随机缩放
_C.AUGMENTATION.SCALE_RANGE = [0.5, 1.5]  # 缩放范围
_C.AUGMENTATION.RANDOM_FLIP = True  # 随机翻转
_C.AUGMENTATION.MOSAIC = True  # 马赛克增强
_C.AUGMENTATION.LSJ = True  # 大尺度抖动增强
_C.AUGMENTATION.SPECKLE_NOISE = True  # 斑点噪声增强
_C.AUGMENTATION.CONTRAST_ADJUST = True  # 对比度调整

# 测试配置
_C.TEST = CN()
_C.TEST.CONF_THRESH = 0.3  # 置信度阈值
_C.TEST.IOU_THRESH = 0.5  # IoU阈值
_C.TEST.RNMS = True  # 是否使用旋转非极大值抑制
_C.TEST.TTA = False  # 是否使用测试时增强

# 评估指标配置
_C.EVAL = CN()
_C.EVAL.METRICS = ['mAP', 'precision', 'recall', 'FAR', 'FPS']  # 评估指标
_C.EVAL.IOU_THRESHOLDS = [0.5, 0.75, 0.5:0.95:0.05]  # IoU阈值列表

# 设备配置
_C.DEVICE = CN()
_C.DEVICE.GPU_IDS = [0]  # GPU ID列表
_C.DEVICE.DISTRIBUTED = False  # 是否使用分布式训练
_C.DEVICE.MIXED_PRECISION = True  # 是否使用混合精度训练

# 输出配置
_C.OUTPUT = CN()
_C.OUTPUT.DIR = './results'  # 输出目录
_C.OUTPUT.SAVE_MODEL = True  # 是否保存模型
_C.OUTPUT.SAVE_FREQ = 10  # 模型保存频率
_C.OUTPUT.LOG_INTERVAL = 10  # 日志打印间隔
_C.OUTPUT.VIS_INTERVAL = 50  # 可视化间隔

# 优化配置
_C.OPTIMIZATION = CN()
_C.OPTIMIZATION.KNOWLEDGE_DISTILLATION = False  # 是否使用知识蒸馏
_C.OPTIMIZATION.TEACHER_MODEL_PATH = ''  # 教师模型路径
_C.OPTIMIZATION.PRUNING = False  # 是否使用模型剪枝
_C.OPTIMIZATION.QUANTIZATION = False  # 是否使用模型量化
_C.OPTIMIZATION.DOMAIN_ADAPTATION = False  # 是否使用域适应

# 跨域学习配置
_C.DOMAIN_ADAPTATION = CN()
_C.DOMAIN_ADAPTATION.METHOD = 'MSFA'  # 域适应方法
_C.DOMAIN_ADAPTATION.LAMBDA = 0.1  # 域适应损失权重

# 多模态融合配置
_C.MULTI_MODAL = CN()
_C.MULTI_MODAL.ENABLE = False  # 是否启用多模态融合
_C.MULTI_MODAL.MODALITIES = ['SAR', 'Optical']  # 融合的模态

# 解析配置函数
def get_cfg_defaults():
    """返回默认配置"""
    return _C.clone()

def update_cfg(cfg, cfg_file):
    """根据配置文件更新配置"""
    cfg.merge_from_file(cfg_file)
    return cfg

def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description='SAR图像多类别有向目标检测')
    parser.add_argument('--config', type=str, default='', help='配置文件路径')
    parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER,
                        help='覆盖配置文件的选项，格式为: KEY VALUE KEY VALUE...')
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    if args.config:
        cfg = update_cfg(cfg, args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # 创建输出目录
    os.makedirs(cfg.OUTPUT.DIR, exist_ok=True)
    
    # 保存最终配置
    with open(os.path.join(cfg.OUTPUT.DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    
    return cfg

if __name__ == '__main__':
    cfg = parse_args()
    print(cfg)