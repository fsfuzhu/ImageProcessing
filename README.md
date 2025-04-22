# COMP2032 Flower Segmentation Project

## Overview

This repository contains the implementation of an advanced image processing pipeline for flower segmentation, developed as part of the COMP2032 Introduction to Image Processing coursework at the University of Nottingham Malaysia. The project focuses on separating flower regions from the background in images with natural-looking boundaries and performing semantic segmentation for flower classification.

## Project Objectives

1. Design and implement a robust image processing pipeline for flower segmentation that works automatically without manual parameter tuning
2. Create a solution that works effectively across images with varying difficulty levels (easy, medium, hard)
3. Preserve natural-looking flower boundaries using edge feathering techniques
4. Evaluate segmentation performance using appropriate metrics (IoU, Dice coefficient, accuracy)
5. Apply semantic classification to evaluate segmentation quality against ground truth masks

## Key Features

- **LAB Color Space Transformation**: Reduces the impact of illumination variations while emphasizing color differences
- **Multi-level Thresholding**: Combines adaptive, Otsu, and multi-level thresholding for robust segmentation
- **Watershed Segmentation**: Effectively separates connected flower regions using gradient information
- **Edge Feathering**: Creates smooth, natural-looking transitions at flower boundaries instead of hard binary edges
- **Connected Component Analysis**: Removes noise and focuses on the main flower object in each image
- **Parameter Independence**: Works across diverse images without requiring manual parameter adjustments

## Repository Structure

```
├── segmentation/               # Core segmentation modules
│   ├── pipeline.py             # Main segmentation pipeline implementation
│   ├── color_spaces.py         # Color space conversion utilities
│   ├── morphology.py           # Morphological operations for post-processing
│   ├── noise_reduction.py      # Noise reduction filters (Gaussian, bilateral, etc.)
│   └── thresholding.py         # Multiple thresholding implementation methods
├── evaluation/                 # Evaluation tools
│   └── evaluate.py             # Metrics calculation and evaluation logic
├── utils/                      # Utility functions
│   └── image_io.py             # Image input/output operations
├── main.py                     # Main execution script
├── Model_input.py              # Semantic classification evaluation
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Installation

### Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/flower-segmentation.git
   cd flower-segmentation
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The project works with two datasets:

### Dataset 1
Used for developing and testing the segmentation pipeline:

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

### Dataset 2
Used for semantic classification evaluation:

```
Dataset_2/
├── input_images/
└── ground_truths/
```

## Usage

### Running the Segmentation Pipeline

To process all images in the dataset with difficulty levels:

```bash
python main.py --input Dataset_1/input_images --output output --pipeline image-processing-pipeline --using-difficulty
```

To process a flat dataset without difficulty subfolders:

```bash
python main.py --input dataset/images --output output --pipeline image-processing-pipeline
```

### Evaluating Segmentation Results

To evaluate segmentation performance against ground truth:

```bash
python main.py --input Dataset_1/input_images --output output --ground-truth Dataset_1/ground_truths --evaluate --evaluate-output evaluation_results --using-difficulty
```

### Command Line Arguments

- `--input`: Directory containing input images (default: `'Dataset_1/input_images'`)
- `--output`: Directory to save segmentation results (default: `'output'`)
- `--pipeline`: Directory to save intermediate pipeline steps (default: `'image-processing-pipeline'`)
- `--ground-truth`: Directory containing ground truth masks (default: `'Dataset_1/ground_truths'`)
- `--evaluate`: Flag to evaluate segmentation results against ground truth
- `--evaluate-output`: Directory to save evaluation results (default: `'evaluation_output'`)
- `--using-difficulty`: Flag to indicate processing images with difficulty level subfolders

## Image Processing Pipeline

Our segmentation pipeline implements a multi-stage approach:

1. **Color Space Transformation**: Conversion to LAB color space to separate luminance from color information
2. **Contrast Enhancement**: Histogram equalization to improve distinction between flowers and background
3. **Gradient Analysis**: Extraction of gradient information from each LAB channel to identify object boundaries
4. **Noise Reduction**: Two-stage filtering with Gaussian and bilateral filters to preserve important edges
5. **Multi-level Thresholding**: Combination of adaptive, Otsu, and multi-level thresholding for robust segmentation
6. **Watershed Segmentation**: Marker-based watershed algorithm for object separation
7. **Connected Component Analysis**: Extraction of the largest connected component that exceeds a minimum size threshold
8. **Edge Feathering**: Creation of a graduated alpha mask for natural-looking flower boundaries

## Evaluation Metrics

The segmentation is evaluated using:

- **IoU (Intersection over Union)**: Measures the overlap between predicted and ground truth masks
- **Dice Coefficient**: F1 score measuring segmentation precision and recall
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Cosine Similarity**: Similarity between feature vectors of segmented and ground truth images

## Results

Our approach achieves the following performance across difficulty levels:

| Difficulty | IoU   | Dice  | Accuracy | Cosine Similarity |
|------------|-------|-------|----------|-------------------|
| Easy       | 0.85  | 0.92  | 0.95     | 0.92              |
| Medium     | 0.73  | 0.84  | 0.87     | 0.85              |
| Hard       | 0.85  | 0.92  | 0.92     | 0.92              |
| **Average**| **0.81** | **0.89** | **0.91** | **0.90**   |

## Authors

- Group 13 Members
  - Sia Ray Young (hfyrs3@nottingham.edu.my)
  - Wong Xuan Kai (hfyxw3@nottingham.edu.my)
  - Ko Jeng Jun (hfyjk8@nottingham.edu.my)
  - Xia Wen Jun (hcywx1@nottingham.edu.my)
  - Wang Jun Ru (hcyjw2@nottingham.edu.my)

## Acknowledgments

This project was completed as coursework for COMP2032 Introduction to Image Processing at the University of Nottingham Malaysia.

## References

- R.C. Gonzalez and R.E. Woods. (2018). Digital Image Processing. (Fourth Edition). Prentice Hall.
- Chris Solomon and Toby Breckon. (2010). Fundamentals of Digital Image Processing: A Practical Approach with Examples in Matlab. Wiley
- Prateek Joshi. (2015). OpenCV with Python by Example. Packt Publishing
- Sandipan Dey. (2020). Python Image Processing Cookbook. Packt Publishing