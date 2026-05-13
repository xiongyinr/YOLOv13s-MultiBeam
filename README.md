# YOLOv13s-MultiBeam: 基于改进YOLOv13的多波束声呐图像目标检测

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 📋 项目简介

本项目是一个基于YOLOv13的多波束声呐图像目标检测系统，专门针对水下声呐图像的特殊性质进行了深度优化。项目聚焦于解决多波束声呐图像中的**小目标检测**、**高纵横比图像处理**、**灰度结构特征提取**等关键问题。

### 核心特点

- 🎯 **针对性优化**：专门针对多波束声呐灰度图像的特征进行算法改进
- 🔍 **小目标增强**：引入P2检测层，显著提升小目标检测能力
- 🏗️ **结构创新**：浅层特征增强 + 改进的Neck融合机制
- 📊 **损失函数优化**：IoU + NWD联合损失，提升小目标定位稳定性
- 🚀 **推理策略**：支持切片推理和多尺度测试，提高检测精度

### 应用场景

- 水下目标检测与识别
- 多波束声呐图像分析
- 海洋工程监测
- 水下机器人视觉系统

---

## 🗂️ 项目结构

```
YOLOv13s-MultiBeam/
├── yolov13s/                    # YOLOv13核心代码库，基础YOLOv13模型
│   ├── ultralytics/             # Ultralytics框架
│   ├── train.py                 # 训练脚本
│   ├── val.py                   # 验证脚本
│   ├── test.py                  # 测试脚本
│   ├── data.yaml                # 数据集配置
│   └── requirements.txt         # 依赖包列表
├── dataset/                     # 数据集目录
│   ├── train/                   # 训练集
│   │   ├── images/              # 训练图像
│   │   └── labels/              # 训练标签
│   ├── val/                     # 验证集
│   │   ├── images/              # 验证图像
│   │   └── labels/              # 验证标签
│   └── test/                    # 测试集
│       ├── images/              # 测试图像
│       └── labels/              # 测试标签
├── EXP01-yolov13/               # 实验1：添加P2高分辨率层
├── EXP02-yolov13/               # 实验2：在P2高分辨率层基础上添加EAS注意力和WFF加权融合
├── EXP03-yolov13/               # 实验3：构建CIoU与NWD结合的联合损失函数
├── EXP04-app/                   # 实验4：应用部署，封装为APP
├── result/                      # 实验结果目录
│   ├── yolo11s_baseline/        # YOLO11s基线结果
│   ├── yolo13s_baseline/        # YOLOv13s基线结果
│   ├── yolo13s_p2/              # 添加P2层结果
│   ├── yolo13s_p2_eca-wff/      # P2+ECA+WFF结果
│   └── yolo13s_p2_eca-wff-loss/ # 完整改进模型结果
└── README.md                    # 项目说明文档
```

---

## 🎯 数据集说明

### 数据集特征

本项目使用的多波束声呐图像数据集具有以下特点：

- **图像类型**：灰度拉伸图像（stretched），保留原始灰度关系并增强对比度
- **图像尺寸**：高纵横比（如1024×1950、512×1776、1024×1028）
- **目标特征**：小目标、稀疏分布、主要位于图像下半区域
- **背景特征**：背景占比大、纹理较弱、颜色语义缺失

### 类别信息

数据集包含10个目标类别：

| ID | 类别名称 | 英文名称 | 说明 |
|----|---------|---------|------|
| 0 | 立方体 | cube | 方形目标物 |
| 1 | 球体 | ball | 球形目标物 |
| 2 | 圆柱体 | cylinder | 圆柱形目标物 |
| 3 | 人体 | human body | 潜水员等人体目标 |
| 4 | 平面 | plane | 平面结构物 |
| 5 | 圆形网箱 | circle cage | 圆形养殖网箱 |
| 6 | 方形网箱 | square cage | 方形养殖网箱 |
| 7 | 金属桶 | metal bucket | 金属桶状物体 |
| 8 | 轮胎 | tyre | 轮胎类目标 |
| 9 | 水下机器人 | rov | ROV设备 |

### 数据集划分

- **训练集（train）**：用于模型训练
- **验证集（val）**：用于超参数调优和模型选择
- **测试集（test）**：用于最终性能评估

---

## 🔧 环境配置

### 系统要求

- **操作系统**：Windows 10/11 或 Linux
- **Python版本**：3.11
- **CUDA版本**：11.x（推荐11.8）
- **显存要求**：≥8GB（推荐≥12GB用于高分辨率训练）

### 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/yourusername/YOLOv13s-MultiBeam.git
cd YOLOv13s-MultiBeam
```

2. **创建虚拟环境**

```bash
conda create -n yolov13 python=3.11
conda activate yolov13
```

3. **安装依赖**

```bash
cd yolov13s
pip install -r requirements.txt
pip install -e .
```

4. **（可选）安装Flash Attention加速**

```bash
# Linux系统
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### 主要依赖包

```
torch==2.2.2
torchvision==0.17.2
timm==1.0.14
albumentations==2.0.4
opencv-python==4.9.0.80
onnx==1.14.0
onnxruntime-gpu==1.18.0
pycocotools==2.0.7
numpy==1.26.4
```

---

## 🚀 快速开始

### 1. 数据准备

确保数据集按照以下格式组织：

```
dataset/
├── train/
│   ├── images/
│   │   ├── 00001.png
│   │   └── ...
│   └── labels/
│       ├── 00001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

修改 `data.yaml` 配置文件中的路径：

```yaml
train: "path/to/dataset/train/images"
val: "path/to/dataset/val/images"
test: "path/to/dataset/test/images"

nc: 10
names:
  0: cube
  1: ball
  2: cylinder
  3: human body
  4: plane
  5: circle cage
  6: square cage
  7: metal bucket
  8: tyre
  9: rov
```

### 2. 模型训练

#### 基础训练

```python
from ultralytics import YOLO

# 加载模型配置
model = YOLO('yolov13s.yaml')

# 训练模型
results = model.train(
    data='data.yaml',
    epochs=300,
    batch=16,
    imgsz=1024,          # 高分辨率输入
    device='0',
    project='runs/train',
    name='yolov13s_multibeam'
)
```

#### 改进模型训练（推荐）

```python
from ultralytics import YOLO

# 加载改进的模型配置
model = YOLO('yolov13s_p2_enhanced.yaml')

# 训练参数
results = model.train(
    data='data.yaml',
    epochs=200,
    batch=16,
    imgsz=1024,
    scale=0.9,           # 数据增强尺度
    mosaic=0.5,          # 轻量Mosaic增强
    mixup=0.0,           # 关闭MixUp
    copy_paste=0.1,      # 轻量Copy-Paste
    hsv_h=0.0,           # 关闭HSV色调增强
    hsv_s=0.0,           # 关闭HSV饱和度增强
    hsv_v=0.2,           # 轻量亮度增强
    device='0',
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
    project='runs/train',
    name='yolov13s_improved'
)
```

### 3. 模型验证

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/yolov13s_improved/weights/best.pt')

# 验证模型
metrics = model.val(
    data='data.yaml',
    imgsz=1024,
    batch=8,
    device='0'
)

# 打印评估指标
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

### 4. 模型推理

#### 单张图像推理

```python
from ultralytics import YOLO

model = YOLO('runs/train/yolov13s_improved/weights/best.pt')

# 推理
results = model.predict(
    source='path/to/image.png',
    imgsz=1024,
    conf=0.25,
    iou=0.45,
    device='0',
    save=True
)

# 显示结果
results[0].show()
```

#### 批量推理

```python
from ultralytics import YOLO

model = YOLO('runs/train/yolov13s_improved/weights/best.pt')

# 批量推理
results = model.predict(
    source='dataset/test/images',
    imgsz=1024,
    conf=0.25,
    iou=0.45,
    device='0',
    save=True,
    project='runs/predict',
    name='test_results'
)
```

#### 切片推理（高精度模式）

```python
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 加载模型
model_path = 'runs/train/yolov13s_improved/weights/best.pt'

# SAHI切片推理
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.25,
    device='cuda:0'
)

# 执行切片推理
result = get_sliced_prediction(
    'path/to/image.png',
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# 导出结果
result.export_visuals(export_dir='runs/sahi_results/')
```

### 5. 模型导出

```python
from ultralytics import YOLO

model = YOLO('runs/train/yolov13s_improved/weights/best.pt')

# 导出为ONNX格式
model.export(format='onnx', dynamic=True, simplify=True)

# 导出为TensorRT格式（需要TensorRT环境）
model.export(format='engine', half=True, workspace=4)
```

---

## 🎨 核心改进方案

### 改进路线图

本项目针对多波束声呐图像的特殊性质，提出了系统化的改进方案：

```
输入层优化 → Backbone增强 → Neck改进 → 损失函数优化 → 推理策略优化
```

### 1. 输入层优化

#### 高分辨率输入
- **改进内容**：将输入尺寸从640提升至1024/1280
- **改进原因**：多波束声呐图像纵横比大，目标占比小，高分辨率可保留更多细节
- **预期收益**：小目标可分辨性提升15-20%

#### 纵横比保持
- **改进内容**：训练时保持原始图像纵横比，避免过度压缩
- **改进原因**：减少目标形变，保持真实形态特征
- **预期收益**：定位精度提升5-10%

### 2. Backbone改进

#### 浅层特征增强
- **改进内容**：在Backbone浅层（Stage1/Stage2）引入ECA注意力机制
- **改进原因**：声呐图像判别信息主要来自边缘和形态，浅层特征至关重要
- **技术细节**：
  ```python
  # ECA模块插入位置
  Backbone:
    - Stage1 → Conv → ECA → Output
    - Stage2 → C2f → ECA → Output
  ```
- **预期收益**：边缘检测能力提升，小目标召回率提升8-12%

#### 灰度结构适配
- **改进内容**：提供1通道输入版本作为对照实验
- **改进原因**：声呐图像无颜色语义，单通道可减少冗余计算
- **预期收益**：参数量减少约30%，推理速度提升10-15%

### 3. Neck改进

#### P2小目标检测层
- **改进内容**：增加P2检测层（stride=4），保留更高分辨率特征
- **改进原因**：目标平均占比仅0.58%-0.87%，P3层（stride=8）过于粗糙
- **网络结构**：
  ```
  原始：P3(80×80) → P4(40×40) → P5(20×20)
  改进：P2(160×160) → P3(80×80) → P4(40×40) → P5(20×20)
  ```
- **预期收益**：小目标AP提升10-15%，这是最关键的改进

#### 高分辨率特征融合
- **改进内容**：在Neck融合节点引入Coordinate Attention
- **改进原因**：长条形图像需要更好的空间方向建模
- **预期收益**：多尺度融合质量提升，整体mAP提升3-5%

### 4. 损失函数优化

#### IoU + NWD联合损失
- **改进内容**：
  ```python
  L_box = λ₁ * L_IoU + λ₂ * L_NWD
  ```
  其中：
  - L_IoU：WIoU或CIoU
  - L_NWD：Normalized Wasserstein Distance
  - λ₁=1.0, λ₂=0.5（推荐值）

- **改进原因**：
  - 小目标框轻微偏移导致IoU剧烈波动
  - NWD从分布距离角度衡量，对小目标更稳定

- **数学表达**：
  ```
  NWD(B₁, B₂) = exp(-√(W₂(N₁, N₂))/C)
  ```
  其中W₂为2-Wasserstein距离，C为归一化常数

- **预期收益**：小目标定位稳定性提升，训练收敛更快

#### Focal Loss分类损失
- **改进内容**：使用Focal Loss替代BCE Loss
- **改进原因**：背景占比大，正负样本严重不平衡
- **预期收益**：减少易分类背景样本的主导作用，精度提升2-3%

### 5. 数据增强策略

#### 灰度结构保持型增强
- **推荐增强**：
  - ✅ 轻量亮度/对比度扰动
  - ✅ 轻量Gamma扰动
  - ✅ 轻量高斯噪声
  - ✅ 水平翻转
  - ✅ 目标框上下文裁剪
- **改进原因**：声呐图像无颜色语义，过强增强会引入伪特征

---

## 📊 实验结果

### 

实验结果保存在 `result/` 目录下，包含：
- 训练曲线（loss、mAP、precision、recall）
- 混淆矩阵
- PR曲线
- 检测结果可视化

---

## 📝 实验配置

### 训练配置

```yaml
# 基础配置
epochs: 200
batch_size: 8
imgsz: 640
device: '0'

# 优化器配置
optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005

# 学习率调度
warmup_epochs: 5
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# 数据增强
scale: 0.9
mosaic: 0.5
mixup: 0.0
copy_paste: 0.1
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.2
degrees: 0.0
translate: 0.1
flipud: 0.0
fliplr: 0.5

# 损失权重
box: 7.5
cls: 0.5
dfl: 1.5
```

### 推理配置

```yaml
# 基础推理
imgsz: 1024
conf: 0.25
iou: 0.45
max_det: 300

# 切片推理
slice_height: 640
slice_width: 640
overlap_height_ratio: 0.2
overlap_width_ratio: 0.2

# 多尺度测试
scales: [960, 1024, 1280]
```

### 

### 

---

## 🤝 贡献指南

欢迎对本项目提出改进建议或贡献代码！

### 贡献方式

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交Pull Request

### 问题反馈

如果您在使用过程中遇到问题，请通过以下方式反馈：
- 提交Issue
- 发送邮件至：renxiongyin@gmail.com

---

## 📄 许可证

本项目基于MIT许可证开源，详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 感谢 [Ultralytics](https://github.com/ultralytics/ultralytics) 提供的优秀YOLO框架
- 感谢 [YOLOv13](https://github.com/iMoonLab/yolov13) 团队的开源工作
- 感谢[海洋空间环境感知开源计划]([1]   https://figshare.com/articles/dataset/UATD_Dataset/21331143/3)
- 感谢束远明老师本科四年来的照顾与教导
- 感谢所有为本项目提供帮助和建议的老师和同学

---

## 📧 联系方式

- **作者**：DOULA
- **邮箱**：renxiongyin@gmail.com
- **GitHub**：https://github.com/xiongyinr

---

## 📌 更新日志

### v1.0.0 (2026-05-13)
- ✨ 初始版本发布
- ✨ 实现基于YOLOv13的多波束声呐图像检测
- ✨ 添加P2小目标检测层
- ✨ 集成ECA注意力机制
- ✨ 实现IoU+NWD联合损失
- ✨ 支持切片推理和多尺度测试

---

<div align="center">

**如果这个项目对您有帮助，请给个⭐Star支持一下！**

Made with ❤️ for Underwater Object Detection

</div>
