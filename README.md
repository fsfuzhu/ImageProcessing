# 花朵分割与分类系统

本项目是COMP2032 (Introduction to Image Processing) 课程的作业，实现了一个完整的花朵分割和分类系统。该系统可以从图像中分割出花朵，并使用语义分割方法对它们进行分类。

## 项目结构

```
flower_segmentation_classification/
│
├── main.py                        # 主程序入口
├── config.py                      # 配置文件
├── transformer_integration.py     # Transformer模型集成
├── semantic_segmentation.py       # 语义分割整合
├── requirements.txt               # 依赖包列表
├── README.md                      # 项目说明文档
│
├── segmentation/                  # 分割相关模块
│   ├── __init__.py
│   ├── color_space.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   └── postprocessing.py
│
├── evaluation/                    # 评估相关模块
│   ├── __init__.py
│   ├── metrics.py
│   └── visualization.py
│
└── utils/                         # 通用工具函数
    ├── __init__.py
    ├── file_io.py
    └── image_utils.py
```

## 安装依赖

本项目依赖于以下Python库：

```
opencv-python>=4.5.5
numpy>=1.23.5
matplotlib>=3.4.0
scikit-learn>=1.1.1
scikit-image>=0.19.0
tqdm>=4.62.0
```

可以使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python main.py --input_dir <输入图像目录> --ground_truth_dir <真实掩码目录> --output_dir <输出目录>
```

### 参数说明

- `--input_dir`: 输入图像目录，默认为 'input-image'
- `--ground_truth_dir`: 真实掩码目录，默认为 'ground-truth'
- `--output_dir`: 输出目录，默认为 'output'
- `--pipeline_dir`: 处理流程可视化目录，默认为 'image-processing-pipeline'
- `--difficulty`: 处理的图像难度，可选 'all', 'easy', 'medium', 'hard'，默认为 'all'
- `--eval`: 是否评估分割结果，默认为 False
- `--visualize`: 是否保存处理流程的可视化结果，默认为 False

### 目录结构要求

输入图像和真实掩码应按以下结构组织：

```
input-image/
│── easy/
│   ├── image_001.jpg
│   ├── image_002.jpg
│── medium/
│   ├── image_003.jpg
│   ├── image_004.jpg
│── hard/
│   ├── image_005.jpg
│   └── image_006.jpg

ground-truth/
│── easy/
│   ├── image_001.png
│   ├── image_002.png
│── medium/
│   ├── image_003.png
│   ├── image_004.png
│── hard/
│   ├── image_005.png
│   └── image_006.png
```

## 图像处理流程

本系统的图像处理流程包含以下主要步骤：

1. **预处理**: 图像尺寸调整、去噪、对比度增强等
2. **颜色空间转换**: 将图像转换到合适的颜色空间（如HSV, LAB）以便更好地分割
3. **分割**: 使用多种方法（阈值分割、颜色聚类、边缘检测等）分割花朵
4. **后处理**: 形态学操作、连通区域分析、边界平滑等改进分割结果
5. **评估**: 计算IoU、Dice系数等评估指标

## 配置调整

可以在 `config.py` 文件中调整各个处理步骤的参数，例如：

- 预处理参数（图像大小、去噪方法等）
- 颜色空间选择
- 分割算法参数
- 后处理参数

## 评估指标

系统支持多种评估指标，包括：

- IoU (Intersection over Union)
- Dice系数
- 精确率和召回率
- 边界F1分数

## 示例结果

运行示例：

```bash
python main.py --input_dir samples/input --ground_truth_dir samples/ground_truth --output_dir results --eval --visualize
```

## 语义分割与分类

完成花朵分割后，系统使用提供的Transformer模型对分割的花朵进行分类。分类结果以余弦相似度和准确率进行评估。

## 注意事项

- 确保输入图像和真实掩码的文件名相匹配
- 系统默认处理.jpg和.png格式的图像
- 评估模式需要真实掩码数据