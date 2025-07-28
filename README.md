# 🚀 YOLOv5 自定义目标检测模板


这是一个基于官方 **YOLOv5** 的增强型项目模板，旨在帮助您以前所未有的速度构建、训练和部署高性能的自定义目标检测模型。无论您是从事工业质检、安防监控、自动驾驶还是创意应用，此模板都能提供一个坚实且高效的起点。

![目标检测示例](https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg)

## ✨ 项目亮点

- 🔧 **一键启动**: 提供了预设的目录结构和脚本，真正实现开箱即用。
- 🧠 **迁移学习**: 无缝加载官方预训练权重，极大缩短您的训练时间。
- 📊 **全面评估**: 内置 mAP 计算、PR 曲线、混淆矩阵等多种评估工具，全方位掌握模型性能。
- 📱 **多平台部署**: 支持对图像、视频、网络摄像头进行推理，并可轻松导出为 ONNX、TensorRT 等格式。
- ⚡ **性能优化**: 集成了模型量化、TensorRT 加速等业界领先的优化方案。
- 🔄 **持续集成**: 包含 GitHub Actions 示例，助您轻松实现自动化训练与测试。

## 📋 环境要求

在开始之前，请确保您的环境中已安装以下软件：
- **Python**: 3.8 或更高版本
- **Git**: 用于版本控制和克隆仓库
- **PyTorch**: 1.7 或更高版本
- 强烈建议使用配备 **NVIDIA GPU** 的环境以获得最佳训练性能。

## 🏁 快速开始

#### 1. 克隆项目仓库

```bash
# 克隆主项目
git clone https://github.com/[Your-GitHub-Username]/my-yolov5-project.git
cd my-yolov5-project
```

#### 2. 初始化 YOLOv5 子模块

我们使用 `git submodule` 来引用官方 YOLOv5 仓库，确保代码同步和纯净。

```bash
# 初始化并拉取子模块代码
git submodule update --init --recursive
```

#### 3. 创建虚拟环境并安装依赖

> **最佳实践**: 始终使用虚拟环境（如 venv 或 conda）来隔离项目依赖。

```bash
# 创建并激活虚拟环境 (Linux/macOS)
python -m venv .venv
source .venv/bin/activate

# (Windows 使用)
# .venv\Scripts\activate

# 安装所有必需的依赖包
pip install -r yolov5/requirements.txt
```

#### 4. 验证安装

运行一个简单的检测任务来验证 YOLOv5 是否已正确安装。

```bash
python yolov5/detect.py --weights yolov5s.pt --source yolov5/data/images/bus.jpg
```
> **预期结果**: 程序会自动下载 `yolov5s.pt` 权重，并对 `bus.jpg` 图像进行检测。检测结果将保存在 `runs/detect/exp` 目录下。

## 📂 数据集准备

#### 目录结构

一个结构良好的数据集是成功的一半。请遵循以下结构组织您的文件：

```
datasets/my_dataset/
├── train/
│   ├── images/  # 训练图像 (.jpg, .png, etc.)
│   └── labels/  # YOLO 格式的标签文件 (.txt)
├── val/
│   ├── images/  # 验证图像
│   └── labels/  # 验证标签
└── test/        # (可选) 测试图像
```
> **提示**: 您可以使用 `LabelImg` 或 `Roboflow` 等工具来标注数据并生成 YOLO 格式的标签。

#### 创建数据集配置文件

在 `datasets/` 目录下创建一个 `.yaml` 文件来描述您的数据集，例如 `my_dataset.yaml`:

```yaml
# 训练集和验证集的路径 (相对于 yolov5 目录)
train: ../datasets/my_dataset/train/images
val: ../datasets/my_dataset/val/images

# 类别数量
nc: 3

# 类别名称 (顺序必须与标签文件中的类别索引对应)
names: ['cone', 'person', 'vehicle']
```

## 🏋️ 模型训练

使用我们提供的 `scripts/train.py` 脚本（或直接使用 `yolov5/train.py`）开始训练。

#### 从预训练权重开始 (推荐)

```bash
python yolov5/train.py \
  --weights yolov5s.pt \
  --data datasets/my_dataset.yaml \
  --epochs 100 \
  --batch-size 16 \
  --img-size 640 \
  --name my_first_run
```
> **说明**:
> *   `--weights`: `yolov5s.pt` 表示从小型预训练模型开始训练。您也可以选择 `yolov5m.pt`, `yolov5l.pt` 等。
> *   `--batch-size`: 根据您的 GPU 显存大小进行调整。如果遇到显存不足 (CUDA out of memory) 的错误，请减小此值。
> *   `--name`: 本次实验的名称，所有结果将保存在 `runs/train/my_first_run` 目录下。

#### 监控训练过程

YOLOv5 与 TensorBoard 深度集成。在训练开始后，运行以下命令：

```bash
tensorboard --logdir runs/train
```
然后在浏览器中打开 `http://localhost:6006`，即可实时查看损失函数、mAP 等指标的变化。

## 📈 模型评估

训练完成后，使用 `val.py` 脚本在验证集上评估模型的最终性能。

```bash
python yolov5/val.py \
  --weights runs/train/my_first_run/weights/best.pt \
  --data datasets/my_dataset.yaml \
  --img-size 640 \
  --task val
```
> **提示**: `best.pt` 是在验证集上取得最佳 mAP 的模型权重。

#### 核心评估指标

| 指标 | 描述 | 参考目标 |
| :--- | :--- | :--- |
| **mAP@0.5** | IoU 阈值为 0.5 时的平均精度，是衡量模型好坏的主要标准。 | > 0.8 |
| **mAP@.5:.95**| 在 IoU 阈值从 0.5 到 0.95 范围内，以 0.05 为步长计算的 mAP 均值。 | > 0.6 |
| **Precision** | 精确率 (查准率)，表示预测为正的样本中有多少是真正的正样本。 | > 0.85 |
| **Recall** | 召回率 (查全率)，表示所有正样本中有多少被成功预测。 | > 0.8 |

## 💡 模型推理

使用您训练好的模型对新数据进行预测。

#### 图像检测

```bash
python yolov5/detect.py \
  --weights runs/train/my_first_run/weights/best.pt \
  --source path/to/your/test_images/ \
  --conf-thres 0.4
```

#### 视频检测

```bash
python yolov5/detect.py \
  --weights runs/train/my_first_run/weights/best.pt \
  --source path/to/your/video.mp4
```

#### 实时摄像头检测

```bash
python yolov5/detect.py \
  --weights runs/train/my_first_run/weights/best.pt \
  --source 0  # 0 代表默认摄像头
```

## 🚀 性能优化与部署

为了在生产环境中获得最佳性能，您需要对模型进行优化和导出。

使用 `yolov5/export.py` 脚本可以将 `.pt` 模型转换为不同格式：

```bash
python yolov5/export.py \
  --weights runs/train/my_first_run/weights/best.pt \
  --img-size 640 \
  --include onnx engine # 同时导出 ONNX 和 TensorRT engine
```

#### 优化策略概览

1.  **ONNX**: 一种开放的神经网络交换格式，具有良好的跨平台兼容性。
2.  **TensorRT (推荐)**: NVIDIA 的高性能深度学习推理引擎，可为 NVIDIA GPU 提供极致的推理速度。
3.  **INT8 量化**: 将模型权重从 FP32（32位浮点数）转换为 INT8（8位整数），可显著减小模型体积并加速计算，但可能伴有轻微的精度损失。

## 🔧 高级功能

#### 自定义模型架构

在 `models/` 目录下创建一个新的 `.yaml` 文件（例如 `my_custom_model.yaml`），修改网络结构（如深度、宽度、模块），然后在训练时通过 `--cfg` 参数指定。

```bash
python yolov5/train.py --cfg models/my_custom_model.yaml ...
```

#### 迁移学习与冻结训练

如果你想在一个已有的自定义模型上继续训练，可以冻结部分层级的权重。

```bash
python yolov5/train.py \
  --weights path/to/your/last.pt \
  --data datasets/new_dataset.yaml \
  --epochs 50 \
  --freeze 10  # 冻结骨干网络的前10层
```

## 🤝 贡献指南

我们欢迎任何形式的贡献！如果您有好的想法或发现了问题，请遵循以下流程：

1.  **Fork** 本项目仓库。
2.  创建一个新的功能分支 (`git checkout -b feature/your-amazing-feature`)。
3.  提交您的代码更改 (`git commit -m 'Add some amazing feature'`)。
4.  将您的分支推送到 GitHub (`git push origin feature/your-amazing-feature`)。
5.  创建一个 **Pull Request**。

## 📜 许可证

本项目基于 [MIT 许可证](LICENSE) 发布。

---
 
> **最后更新**: 2025年07月28日  
> **YOLOv5 版本**: v7.0 (或更高)
