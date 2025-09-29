# 大规模SAR图像多类别有向目标检测系统

## 项目概述

这是一个基于深度学习的SAR图像多类别有向目标检测系统，能够自动识别和定位SAR图像中的多种目标，并输出带有旋转角度的边界框。本系统结合了CNN和Transformer的优势，特别优化了SAR图像的特性，能够有效处理相干斑噪声、低对比度和目标方向变化等挑战。

## 主要特性

- **混合CNN-Transformer架构**：结合CSPDarknet和Swin Transformer的优势，同时捕获局部和全局特征
- **多尺度特征融合**：使用FPN-PAN结构进行高效的多尺度特征融合
- **有向目标检测**：支持旋转边界框的检测和回归，适应SAR图像中目标的不同朝向
- **注意力增强**：集成CBAM注意力模块，提高特征表示能力
- **鲁棒的数据增强**：包含马赛克、大尺度抖动、斑点噪声等SAR图像特定的数据增强方法
- **分布式训练支持**：支持多GPU并行训练，加速模型收敛
- **混合精度训练**：支持自动混合精度训练，减少显存占用，提高训练速度

## 项目结构

```
├── config.py              # 配置文件，包含所有超参数和设置
├── data_loader.py         # 数据加载和预处理模块
├── model.py               # 模型定义，包含骨干网络、颈部网络和检测头等
├── utils.py               # 辅助函数，包含数据处理、后处理、评估指标等
├── train.py               # 训练脚本
├── test.py                # 测试脚本
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明文档
```

## 安装说明

### 环境要求

- Python 3.7+ 
- PyTorch 1.8+ 
- CUDA 10.2+ (如需GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集准备

本系统支持自定义SAR图像数据集，数据集的组织结构如下：

```
dataset/
├── images/                # 图像文件夹
│   ├── train/             # 训练集图像
│   ├── val/               # 验证集图像
│   └── test/              # 测试集图像
└── annotations/           # 标注文件夹
    ├── train.json         # 训练集标注文件
    ├── val.json           # 验证集标注文件
    └── test.json          # 测试集标注文件
```

标注文件采用COCO格式，但需要额外添加旋转角度信息。每个边界框的格式为`[x, y, width, height, angle]`，其中`angle`表示旋转角度（度）。

## 配置文件

在训练和测试之前，需要先配置`config.py`文件中的相关参数。主要配置项包括：

- **数据集配置**：数据集路径、类别名称、图像尺寸、批次大小等
- **模型配置**：模型类型、骨干网络、颈部网络、检测头等
- **训练配置**：学习率、优化器、训练轮数、权重衰减等
- **损失函数配置**：各类损失的权重、Focal Loss参数等
- **数据增强配置**：是否启用数据增强、增强方法和参数等
- **测试配置**：置信度阈值、IoU阈值、NMS参数等
- **设备配置**：GPU ID、分布式训练设置、混合精度训练等

## 训练模型

使用以下命令开始训练模型：

```bash
python train.py --config config.py --output_dir output
```

主要参数说明：
- `--config`：配置文件路径，默认为`config.py`
- `--output_dir`：输出目录，用于保存模型和日志，默认为`output`
- `--resume`：可选，恢复训练的模型路径

训练过程中，模型会定期保存到指定的输出目录，并且会在验证集上评估性能，保存性能最佳的模型。

## 测试模型

使用以下命令测试模型性能：

```bash
python test.py --config config.py --model_path output/models/best_model.pth --output_dir output
```

主要参数说明：
- `--config`：配置文件路径，默认为`config.py`
- `--model_path`：模型权重文件路径，必须指定
- `--output_dir`：输出目录，用于保存测试结果，默认为`output`
- `--test_data`：可选，测试数据集路径，覆盖配置文件中的设置

测试完成后，系统会生成以下结果：
- 检测结果文件（保存在`output/results/detections/`目录下）
- 可视化结果图像（保存在`output/results/visualizations/`目录下）
- 性能评估指标（保存在`output/results/metrics.txt`文件中）

## 评估指标

系统支持以下评估指标：

- **mAP (mean Average Precision)**：平均精确率
- **Precision**：精确率
- **Recall**：召回率
- **F1 Score**：F1分数
- **FPS (Frames Per Second)**：每秒处理帧数

## 单张图像推理

可以使用`test.py`中的`inference_single_image`函数对单张图像进行推理。示例代码如下：

```python
import cv2
import torch
from config import Config
from model import SAROrientedDetector
import utils

# 加载配置
config = Config()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = SAROrientedDetector(
    num_classes=config.DATASET.NUM_CLASSES,
    in_channels=config.MODEL.IN_CHANNELS
)

# 加载模型权重
model.load_state_dict(torch.load('output/models/best_model.pth', map_location=device)['model_state_dict'])
model.to(device)
model.eval()

# 读取图像
image = cv2.imread('path/to/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 推理
with torch.no_grad():
    detections = utils.inference_single_image(model, image, config, device)

# 可视化结果
visualized_image = utils.visualize_detections(
    image, 
    detections, 
    config.DATASET.CLASSES, 
    score_threshold=config.TEST.CONFIDENCE_THRESHOLD
)

# 显示结果
cv2.imshow('Detection Results', cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 高级功能

### 分布式训练

系统支持分布式训练，可以在多GPU环境下加速训练过程。使用以下命令启动分布式训练：

```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py --config config.py --output_dir output
```

其中`--nproc_per_node`参数指定使用的GPU数量。

### 混合精度训练

系统支持自动混合精度训练，可以减少显存占用并提高训练速度。在`config.py`中设置：

```python
config.TRAIN.MIXED_PRECISION = True
```

### 知识蒸馏

系统支持知识蒸馏训练，可以将大型模型的知识转移到小型模型中。在`config.py`中设置：

```python
config.OPTIMIZATION.KNOWLEDGE_DISTILLATION.ENABLE = True
config.OPTIMIZATION.KNOWLEDGE_DISTILLATION.TEACHER_MODEL_PATH = 'path/to/teacher/model.pth'
config.OPTIMIZATION.KNOWLEDGE_DISTILLATION.TEMPERATURE = 2.0
config.OPTIMIZATION.KNOWLEDGE_DISTILLATION.ALPHA = 0.5
```

## 注意事项

1. SAR图像通常具有相干斑噪声，建议在预处理阶段使用适当的滤波方法
2. 训练SAR目标检测模型需要大量标注数据，如果数据量有限，可以考虑数据增强和迁移学习
3. 有向目标检测的评估需要使用旋转IoU计算和旋转NMS，系统已内置相关实现
4. 对于大规模数据集，建议使用分布式训练和混合精度训练以提高效率

## 参考资料

1. 技术方案文档：T001-技术方案-大规模SAR图像多类别有向目标检测.md
2. 研究报告：大规模SAR图像多类别有向目标检测研究报告.md

## 联系方式

如有任何问题或建议，请随时联系项目维护者。