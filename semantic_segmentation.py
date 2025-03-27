#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语义分割模块，整合分割流程和Transformer分类器
"""

import os
import argparse
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformer_integration import TransformerClassifier
from utils.file_io import ensure_dir, get_filename, save_metrics
from segmentation.preprocessing import preprocess_image
from segmentation.color_space import convert_color_space
from segmentation.segmentation import segment_flower
from segmentation.postprocessing import postprocess_mask


def process_image_batch(input_dir, output_dir, config, model_path=None):
    """
    批量处理图像，进行分割和分类
    
    参数:
        input_dir: 输入图像目录
        output_dir: 输出目录
        config: 分割配置
        model_path: Transformer模型路径
        
    返回:
        处理结果的字典
    """
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 获取输入图像列表
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_paths.extend(glob.glob(os.path.join(input_dir, "*.png")))
    
    # 初始化分类器 (如果提供了模型路径)
    classifier = None
    if model_path and os.path.exists(model_path):
        classifier = TransformerClassifier(model_path=model_path)
    
    # 处理结果
    results = []
    
    # 处理每张图像
    for image_path in tqdm(image_paths):
        # 获取文件名
        filename = get_filename(image_path)
        output_path = os.path.join(output_dir, f"{filename}.jpg")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue
        
        # 图像预处理
        preprocessed = preprocess_image(image, config.preprocessing)
        
        # 颜色空间转换
        converted = convert_color_space(preprocessed, config.color_space.space)
        
        # 分割
        segmented, masks, features = segment_flower(converted, preprocessed, config.segmentation)
        
        # 后处理
        final_mask = postprocess_mask(masks, features, config.postprocessing)
        
        # 应用掩码到原始图像，创建分割结果
        segmented_flower = cv2.bitwise_and(image, image, mask=final_mask)
        
        # 创建黑色背景
        black_background = np.zeros_like(image)
        
        # 将分割的花朵放到黑色背景上
        output = cv2.bitwise_or(black_background, segmented_flower)
        
        # 保存结果
        cv2.imwrite(output_path, output)
        
        # 如果有分类器，提取特征并保存
        if classifier:
            try:
                features = classifier.extract_features(output)
                results.append({
                    'filename': filename,
                    'features': features
                })
            except Exception as e:
                print(f"Error extracting features for {filename}: {str(e)}")
    
    return results


def evaluate_dataset(segmented_dir, ground_truth_dir, model_path, threshold=0.9):
    """
    评估分割和分类结果
    
    参数:
        segmented_dir: 分割结果目录
        ground_truth_dir: 真实掩码目录
        model_path: Transformer模型路径
        threshold: 分类准确率阈值
        
    返回:
        评估结果
    """
    # 初始化分类器
    classifier = TransformerClassifier(model_path=model_path)
    
    # 评估数据集
    metrics_df, overall_similarity, overall_accuracy = classifier.evaluate_dataset(
        segmented_dir,
        ground_truth_dir,
        threshold=threshold
    )
    
    return metrics_df, overall_similarity, overall_accuracy


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Semantic Segmentation of Flowers')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output segmented images')
    parser.add_argument('--ground_truth_dir', type=str,
                        help='Directory containing ground truth masks for evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to Transformer model weights')
    parser.add_argument('--results_csv', type=str, default='semantic_results.csv',
                        help='CSV file to save results')
    args = parser.parse_args()
    
    # 导入配置
    from config import Config
    config = Config()
    
    print("Starting segmentation process...")
    
    # 处理图像
    process_image_batch(args.input_dir, args.output_dir, config, model_path=args.model_path)
    
    print("Segmentation completed. Starting evaluation...")
    
    # 如果提供了真实掩码目录，进行评估
    if args.ground_truth_dir and os.path.exists(args.ground_truth_dir):
        metrics_df, overall_similarity, overall_accuracy = evaluate_dataset(
            args.output_dir,
            args.ground_truth_dir,
            args.model_path
        )
        
        # 保存评估结果
        metrics_df.to_csv(args.results_csv, index=False)
        
        # 显示结果
        print(f"\nOverall Segmentation Accuracy: {overall_accuracy:.2f}%")
        print(f"Overall Cosine Similarity: {overall_similarity:.2f}%")
    else:
        print("Ground truth directory not provided or does not exist. Skipping evaluation.")
    
    print("Semantic segmentation process completed.")


if __name__ == "__main__":
    main()