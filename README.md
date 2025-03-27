# 花朵分割与分类系统

本项目是COMP2032 (Introduction to Image Processing) 课程的作业，实现了一个完整的花朵分割和分类系统。该系统使用掩码从图像中分割出花朵，并使用语义分割方法对它们进行分类。

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
opencv-python
numpy
matplotlib
scikit-learn
scikit-image
torch
torchvision
transformers
tqdm
pandas
Pillow
```

可以使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 处理流程

本项目的处理流程包含以下主要步骤：

1. **掩码提取**: 从ground_truth目录中读取包含红色区域的掩码图像
2. **掩码转换**: 将红色掩码转换为二值掩码（白色为花朵区域，黑色为背景）
3. **花朵提取**: 使用二值掩码从对应的原始图像中提取花朵区域
4. **黑色背景合成**: 将提取的花朵放在黑色背景上生成最终输出

## 使用方法

### 基本使用

```bash
python main.py --input_dir <输入图像目录> --mask_dir <掩码图像目录> --output_dir <输出目录>
```

### 参数说明

- `--input_dir`: 输入图像目录，默认为 'input-image'
- `--mask_dir`: 掩码图像目录，默认为 'ground-truth'
- `--output_dir`: 输出目录，默认为 'output'
- `--single_file`: 处理单个文件而不是整个目录

### 目录结构要求

输入图像和掩码图像应按以下结构组织：

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

## 语义分割部分

提取花朵后，系统使用Transformer模型对分割的花朵进行分类：

1. 首先将分割后的花朵输入到Transformer模型
2. 通过模型提取特征并计算与ground truth的余弦相似度
3. 根据相似度阈值判断分类准确性

运行语义分割评估：

```bash
python semantic_segmentation.py --input_dir <分割结果目录> --ground_truth_dir <真实掩码目录> --model_path <Transformer模型路径>
```

## 示例结果

输入图像：
![输入图像](examples/input.jpg)

掩码图像：
![掩码图像](examples/mask.png)

输出结果：
![输出结果](examples/output.jpg)

## 注意事项

- 确保输入图像和掩码图像的文件名相匹配
- 掩码图像中的红色区域将被识别为花朵区域
- 系统默认处理.jpg和.png格式的图像
- 评估模式需要真实掩码数据和Transformer模型