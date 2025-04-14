# COMP2032 Flower Segmentation Project

## Overview

This repository contains the implementation of an advanced image processing pipeline for flower segmentation, developed as part of the COMP2032 Introduction to Image Processing coursework at the University of Nottingham Malaysia. The project focuses on separating flower regions from the background in images and performing semantic segmentation.

## Project Objectives

1.  Design and implement a robust image processing pipeline for flower segmentation.
2.  Create a solution that works across images with varying difficulty levels.
3.  Evaluate segmentation performance using appropriate metrics.
4.  Apply semantic classification to differentiate flower species.

## Repository Structure

```
├── segmentation/             # Core segmentation modules
│   ├── pipeline.py           # Main segmentation pipeline
│   ├── color_spaces.py       # Color space conversion utilities
│   ├── morphology.py         # Morphological operations
│   ├── noise_reduction.py    # Noise reduction filters
│   └── thresholding.py       # Thresholding algorithms
├── evaluation/               # Evaluation tools
│   └── evaluate.py           # Metrics calculation and evaluation
├── utils/                    # Utility functions
│   └── image_io.py           # Image input/output operations
├── main.py                   # Main execution script
├── Model_input.py            # Semantic classification model
└── requirements.txt          # Project dependencies
```

## Installation

### Environment

-   Python 3.11.9
-   Dependencies listed in `requirements.txt`

### Setup Steps

1.  Clone this repository:

    ```bash
    git clone [https://github.com/fsfuzhu/ImageProcessing.git](https://github.com/fsfuzhu/ImageProcessing.git)
    cd ImageProcessing
    ```

2.  Create and activate a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset Structure

The project expects the following dataset structure:

```
Dataset_1/
├── input_images/
│   ├── easy/
│   ├── medium/
│   └── hard/
└── ground_truths/
    ├── easy/
    ├── medium/
    └── hard/
```

## Usage

### Running the Segmentation Pipeline

To process all images in the dataset:

```bash
python main.py --input Dataset_1/input_images --output output --pipeline image-processing-pipeline
```

### Evaluating Segmentation Results

To evaluate segmentation performance against ground truth:

```bash
python main.py --input Dataset_1/input_images --output output --ground-truth Dataset_1/ground_truths --evaluate --evaluate-output evaluation_output
```

### Command Line Arguments

-   `--input`: Directory containing input images (default: `'Dataset_1/input_images'`)
-   `--output`: Directory to save segmentation results (default: `'output'`)
-   `--pipeline`: Directory to save intermediate pipeline steps (default: `'image-processing-pipeline'`)
-   `--ground-truth`: Directory containing ground truth masks (default: `'Dataset_1/ground_truths'`)
-   `--evaluate`: Evaluate segmentation results against ground truth
-   `--evaluate-output`: Directory to save evaluation results (default: `'evaluation_output'`)

## Image Processing Pipeline

Our segmentation pipeline implements a multi-stage approach:

1.  **Color Space Transformation**: Conversion to LAB color space to separate luminance from color information.
2.  **Gradient Calculation**: Extraction of gradient information from each color channel.
3.  **Noise Reduction**: Application of Gaussian and bilateral filtering to reduce noise while preserving edges.
4.  **Multi-level Thresholding**: Combination of adaptive thresholding, Otsu's method, and multi-level thresholding.
5.  **Watershed Segmentation**: Marker-based watershed algorithm for object separation.
6.  **Morphological Operations**: Operations to fill holes and extract the largest connected component.
7.  **Edge Feathering**: Creation of smooth transitions at object boundaries for natural-looking segmentation.

## Semantic Classification

The semantic part uses a vision transformer model (ViT) to perform pixel-wise classification. The model evaluates segmentation quality by comparing against ground truth masks using cosine similarity and accuracy metrics.

## Evaluation Metrics

The segmentation is evaluated using:

-   **IoU (Intersection over Union)**: Measures the overlap between predicted and ground truth masks.
-   **Accuracy**: Pixel-wise classification accuracy.
-   **Dice Coefficient**: F1 score measuring segmentation precision and recall.

## Dependencies

-   numpy
-   opencv-python
-   scikit-image
-   scikit-learn
-   matplotlib
-   pandas
-   torch
-   torchvision

## Authors

-   Group 13 Members
    -   Sia Ray Young
    -   Wong Xuan Kai
    -   Ko Jeng Jun
    -   Xia Wen Jun
    -   Wang Jun Ru
