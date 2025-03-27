#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主程序入口，处理命令行参数，执行分割与分类流程
"""

import os
import argparse
import glob
import cv2
import time
from tqdm import tqdm

from config import Config
from utils.file_io import ensure_dir, get_filename
from segmentation.preprocessing import preprocess_image
from segmentation.color_space import convert_color_space
from segmentation.segmentation import segment_flower
from segmentation.postprocessing import postprocess_mask
from evaluation.metrics import calculate_iou
from evaluation.visualization import save_visualization


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Flower Segmentation and Classification')
    parser.add_argument('--input_dir', type=str, default='input-image',
                        help='Directory containing input images')
    parser.add_argument('--ground_truth_dir', type=str, default='ground-truth',
                        help='Directory containing ground truth masks')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output segmented images')
    parser.add_argument('--pipeline_dir', type=str, default='image-processing-pipeline',
                        help='Directory to save intermediate processing steps')
    parser.add_argument('--difficulty', type=str, choices=['all', 'easy', 'medium', 'hard'], default='all',
                        help='Difficulty level of images to process')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate segmentation against ground truth')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization of segmentation steps')
    
    return parser.parse_args()


def process_image(image_path, config, ground_truth_path=None, visualize=False, pipeline_dir=None):
    """处理单张图像的完整流程"""
    # 读取原始图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, None, None
    
    # 基本信息
    filename = get_filename(image_path)
    
    # 图像预处理
    preprocessed = preprocess_image(image, config.preprocessing)
    
    # 颜色空间转换
    converted = convert_color_space(preprocessed, config.color_space['space'])
    
    # 分割
    segmented, masks, features = segment_flower(converted, preprocessed, config.segmentation)
    
    # 后处理
    final_mask = postprocess_mask(masks, features, config.postprocessing)
    
    # 应用掩码到原始图像
    # Ensure mask is properly formatted before using it
    if final_mask is not None:
        # Convert mask to proper format if needed
        mask_8bit = final_mask.astype('uint8') if final_mask is not None else None
        # Ensure mask has the same dimensions as the image
        if mask_8bit is not None and mask_8bit.shape[:2] != image.shape[:2]:
            mask_8bit = cv2.resize(mask_8bit, (image.shape[1], image.shape[0]))
        # Apply bitwise operation with proper error handling
        try:
            segmented_flower = cv2.bitwise_and(image, image, mask=mask_8bit)
        except cv2.error as e:
            print(f"OpenCV error: {e}")
            # Fallback: Just use the original image if mask fails
            segmented_flower = image.copy()
    else:
        # If mask is None, just use the original image
        segmented_flower = image.copy()
    
    # 创建黑色背景
    black_background = np.zeros_like(image)
    # 将分割的花朵放到黑色背景上
    output = cv2.bitwise_or(black_background, segmented_flower)
    
    # 评估（如果提供了ground truth）
    metrics = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        if ground_truth is not None:
            # 确保二值化
            _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
            metrics = calculate_iou(final_mask, ground_truth)
    
    # 保存中间处理步骤（如果需要）
    if visualize and pipeline_dir:
        image_pipeline_dir = os.path.join(pipeline_dir, filename)
        ensure_dir(image_pipeline_dir)
        save_visualization(image_pipeline_dir, {
            'original': image,
            'preprocessed': preprocessed,
            'color_space': converted,
            'masks': masks,
            'final_mask': final_mask,
            'segmented_flower': segmented_flower,
            'output': output
        })
    
    return output, final_mask, metrics


def main():
    """主函数"""
    args = parse_args()
    config = Config()
    
    # 确保输出目录存在
    ensure_dir(args.output_dir)
    ensure_dir(args.pipeline_dir)
    
    # 准备子目录
    difficulties = ['easy', 'medium', 'hard'] if args.difficulty == 'all' else [args.difficulty]
    
    for diff in difficulties:
        # 确保输出子目录存在
        output_subdir = os.path.join(args.output_dir, diff)
        pipeline_subdir = os.path.join(args.pipeline_dir, diff)
        ensure_dir(output_subdir)
        ensure_dir(pipeline_subdir)
        
        # 获取输入图像列表
        input_subdir = os.path.join(args.input_dir, diff)
        ground_truth_subdir = os.path.join(args.ground_truth_dir, diff)
        
        image_paths = glob.glob(os.path.join(input_subdir, "*.jpg")) + \
                     glob.glob(os.path.join(input_subdir, "*.png"))
        
        print(f"Processing {len(image_paths)} images in '{diff}' category...")
        
        # 处理每张图像
        results = []
        for image_path in tqdm(image_paths):
            filename = get_filename(image_path)
            output_path = os.path.join(output_subdir, f"{filename}.jpg")
            ground_truth_path = None
            
            if args.eval:
                # 尝试查找对应的ground truth图像
                for ext in ['.jpg', '.png']:
                    gt_path = os.path.join(ground_truth_subdir, f"{filename}{ext}")
                    if os.path.exists(gt_path):
                        ground_truth_path = gt_path
                        break
            
            # 处理图像
            output, mask, metrics = process_image(
                image_path, 
                config, 
                ground_truth_path=ground_truth_path, 
                visualize=args.visualize,
                pipeline_dir=pipeline_subdir
            )
            
            if output is not None:
                # 保存结果
                cv2.imwrite(output_path, output)
                
                # 如果有评估指标，保存它们
                if metrics:
                    results.append({
                        'filename': filename,
                        'difficulty': diff,
                        **metrics
                    })
        
        print(f"Completed processing '{diff}' category.")
        
        # 如果有评估结果，计算并显示平均指标
        if results:
            avg_iou = sum(r['iou'] for r in results) / len(results)
            avg_dice = sum(r['dice'] for r in results) / len(results)
            print(f"Average IoU for '{diff}' category: {avg_iou:.4f}")
            print(f"Average Dice for '{diff}' category: {avg_dice:.4f}")
    
    print("All images processed successfully!")


if __name__ == "__main__":
    # 添加对numpy的导入
    import numpy as np
    main()