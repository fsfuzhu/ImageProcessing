#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - 通用版本
花朵分割主脚本
"""

import os
import cv2
import argparse
import glob
import time
from pathlib import Path
import numpy as np

from segmentation.pipeline import FlowerSegmentationPipeline
from evaluation.metrics import calculate_iou, calculate_all_metrics

def ensure_dir(directory):
    """如果目录不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_images(input_dir, output_dir, pipeline_dir, difficulty_levels=None):
    """
    处理输入目录中的所有图像并保存结果
    
    Args:
        input_dir: 包含输入图像的目录
        output_dir: 保存分割输出的目录
        pipeline_dir: 保存中间管线结果的目录
        difficulty_levels: 难度子文件夹名称列表（例如，['easy', 'medium', 'hard']）
    """
    # 初始化分割管线
    pipeline = FlowerSegmentationPipeline()
    
    # 如果没有提供特定的难度级别，则默认处理所有图像
    if difficulty_levels is None:
        difficulty_levels = [""]
    
    total_images = 0
    total_time = 0
    
    # 处理每个难度级别
    for level in difficulty_levels:
        level_input_dir = os.path.join(input_dir, level) if level else input_dir
        level_output_dir = os.path.join(output_dir, level) if level else output_dir
        level_pipeline_dir = os.path.join(pipeline_dir, level) if level else pipeline_dir
        
        # 确保输出目录存在
        ensure_dir(level_output_dir)
        ensure_dir(level_pipeline_dir)
        
        # 查找当前难度级别中的所有图像
        image_paths = glob.glob(os.path.join(level_input_dir, "*.jpg")) + \
                      glob.glob(os.path.join(level_input_dir, "*.png"))
        
        print(f"正在处理 {level_input_dir} 中的 {len(image_paths)} 张图像...")
        
        # 处理每张图像
        for img_path in image_paths:
            img_filename = os.path.basename(img_path)
            img_name = os.path.splitext(img_filename)[0]
            
            print(f"正在处理图像: {img_filename}")
            
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"错误: 无法读取图像 {img_path}")
                continue
            
            # 计时分割过程
            start_time = time.time()
            
            # 应用分割管线
            segmented_img, intermediate_results = pipeline.process(img)
            
            # 记录处理时间
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            total_images += 1
            
            print(f"  处理时间: {elapsed_time:.2f} 秒")
            
            # 保存分割结果
            output_path = os.path.join(level_output_dir, img_filename)
            cv2.imwrite(output_path, segmented_img)
            
            # 保存中间结果
            img_pipeline_dir = os.path.join(level_pipeline_dir, img_name)
            ensure_dir(img_pipeline_dir)
            
            for step_name, result_img in intermediate_results.items():
                step_path = os.path.join(img_pipeline_dir, f"{step_name}.jpg")
                cv2.imwrite(step_path, result_img)
    
    # 打印总体统计信息
    if total_images > 0:
        avg_time = total_time / total_images
        print(f"\n处理了 {total_images} 张图像，耗时 {total_time:.2f} 秒")
        print(f"平均处理时间: {avg_time:.2f} 秒/图像")

def evaluate_segmentation(segmented_dir, ground_truth_dir, difficulty_levels=None):
    """
    评估分割结果与真实标签的对比
    
    Args:
        segmented_dir: 包含分割结果的目录
        ground_truth_dir: 包含真实标签掩码的目录
        difficulty_levels: 要处理的难度级别列表
    """
    if difficulty_levels is None:
        difficulty_levels = [""]
    
    total_metrics = {}
    num_images = 0
    
    print("\n评估分割结果:")
    
    for level in difficulty_levels:
        level_segmented_dir = os.path.join(segmented_dir, level) if level else segmented_dir
        level_gt_dir = os.path.join(ground_truth_dir, level) if level else ground_truth_dir
        
        # 查找所有分割结果
        segmented_paths = glob.glob(os.path.join(level_segmented_dir, "*.jpg")) + \
                         glob.glob(os.path.join(level_segmented_dir, "*.png"))
        
        for seg_path in segmented_paths:
            img_filename = os.path.basename(seg_path)
            gt_path = os.path.join(level_gt_dir, img_filename)
            
            # 如果找不到真实标签则跳过
            if not os.path.exists(gt_path):
                print(f"  警告: 未找到 {img_filename} 的真实标签")
                continue
            
            # 读取分割结果和真实标签
            segmented = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            # 确保二值掩码
            _, segmented = cv2.threshold(segmented, 1, 255, cv2.THRESH_BINARY)
            _, ground_truth = cv2.threshold(ground_truth, 1, 255, cv2.THRESH_BINARY)
            
            # 计算所有指标
            metrics = calculate_all_metrics(segmented, ground_truth)
            
            # 初始化总体指标字典
            if not total_metrics:
                total_metrics = {k: 0 for k in metrics.keys()}
            
            # 累积指标
            for k, v in metrics.items():
                total_metrics[k] += v
            
            print(f"  {img_filename}: IoU = {metrics['iou']:.4f}, Dice = {metrics['dice']:.4f}")
            
            num_images += 1
    
    # 打印总体结果
    if num_images > 0:
        print(f"\n在 {num_images} 张图像上的平均指标:")
        for k, v in total_metrics.items():
            avg_value = v / num_images
            print(f"平均 {k}: {avg_value:.4f}")

def main():
    """运行花朵分割管线的主函数"""
    parser = argparse.ArgumentParser(description="花朵分割管线")
    parser.add_argument('--input', type=str, default='Dataset_1/input_images',
                        help='包含输入图像的目录')
    parser.add_argument('--output', type=str, default='output',
                        help='保存分割结果的目录')
    parser.add_argument('--pipeline', type=str, default='image-processing-pipeline',
                        help='保存中间管线步骤的目录')
    parser.add_argument('--ground-truth', type=str, default='Dataset_1/ground_truths',
                        help='包含真实标签掩码的目录')
    parser.add_argument('--evaluate', action='store_true',
                        help='评估分割结果与真实标签的对比')
    
    args = parser.parse_args()
    
    # 定义难度级别
    difficulty_levels = ['easy', 'medium', 'hard']
    
    # 处理图像
    process_images(args.input, args.output, args.pipeline, difficulty_levels)
    
    # 如果请求则评估结果
    if args.evaluate:
        evaluate_segmentation(args.output, args.ground_truth, difficulty_levels)

if __name__ == "__main__":
    main()