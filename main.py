#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于掩码的花朵提取程序
读取ground_truth中的掩码，提取input_images中对应的花朵
"""

import os
import argparse
import glob
import cv2
import numpy as np
from tqdm import tqdm

from utils.file_io import ensure_dir, get_filename


def extract_red_mask(mask_image):
    """
    从掩码图像中提取红色区域
    
    参数:
        mask_image: 包含红色掩码的图像
        
    返回:
        二值化的掩码（白色为花朵区域，黑色为背景）
    """
    # 转换为HSV颜色空间以更容易检测红色
    hsv = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)
    
    # 定义HSV中的红色范围（红色在HSV中有两个范围）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # 创建红色掩码（包含两个范围）
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 应用形态学操作改善掩码质量
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return red_mask


def process_image_with_mask(image_path, mask_path, output_path):
    """
    使用掩码提取图像中的花朵
    
    参数:
        image_path: 原始图像路径
        mask_path: 掩码图像路径
        output_path: 输出图像保存路径
        
    返回:
        是否成功处理
    """
    # 读取图像和掩码
    image = cv2.imread(image_path)
    mask_image = cv2.imread(mask_path)
    
    if image is None or mask_image is None:
        print(f"Error: Could not read image or mask")
        return False
    
    # 提取红色掩码
    red_mask = extract_red_mask(mask_image)
    
    # 确保掩码和图像尺寸相同
    if red_mask.shape[:2] != image.shape[:2]:
        red_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 应用掩码提取花朵
    segmented_flower = cv2.bitwise_and(image, image, mask=red_mask)
    
    # 创建黑色背景
    black_background = np.zeros_like(image)
    
    # 将分割的花朵放到黑色背景上
    output = cv2.bitwise_or(black_background, segmented_flower)
    
    # 保存结果
    ensure_dir(os.path.dirname(output_path))
    cv2.imwrite(output_path, output)
    
    return True


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Extract flowers using masks')
    parser.add_argument('--input_dir', type=str, default='input-image',
                        help='Directory containing original images')
    parser.add_argument('--mask_dir', type=str, default='ground-truth',
                        help='Directory containing mask images')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output segmented images')
    parser.add_argument('--difficulty', type=str, choices=['all', 'easy', 'medium', 'hard'], default='all',
                        help='Difficulty level of images to process')
    parser.add_argument('--single_file', type=str, default=None,
                        help='Process a single file instead of a directory')
    args = parser.parse_args()
    
    # 处理单个文件的情况
    if args.single_file:
        if not os.path.exists(args.single_file):
            print(f"Error: File {args.single_file} does not exist")
            return
        
        # 获取文件名和对应的掩码文件
        filename = get_filename(args.single_file)
        mask_file = None
        
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = os.path.join(args.mask_dir, f"{filename}{ext}")
            if os.path.exists(mask_path):
                mask_file = mask_path
                break
        
        if mask_file is None:
            print(f"Error: Could not find corresponding mask for {args.single_file}")
            return
        
        # 处理图像
        output_path = os.path.join(args.output_dir, f"{filename}.jpg")
        if process_image_with_mask(args.single_file, mask_file, output_path):
            print(f"Successfully processed {args.single_file} -> {output_path}")
        else:
            print(f"Failed to process {args.single_file}")
        
        return
    
    # 准备子目录
    difficulties = ['easy', 'medium', 'hard'] if args.difficulty == 'all' else [args.difficulty]
    
    for diff in difficulties:
        # 确保输出子目录存在
        output_subdir = os.path.join(args.output_dir, diff)
        ensure_dir(output_subdir)
        
        # 获取掩码文件列表
        mask_subdir = os.path.join(args.mask_dir, diff)
        input_subdir = os.path.join(args.input_dir, diff)
        
        if not os.path.exists(mask_subdir):
            print(f"Warning: Mask directory {mask_subdir} does not exist. Skipping.")
            continue
        
        if not os.path.exists(input_subdir):
            print(f"Warning: Input directory {input_subdir} does not exist. Skipping.")
            continue
        
        mask_files = glob.glob(os.path.join(mask_subdir, "*.png")) + \
                    glob.glob(os.path.join(mask_subdir, "*.jpg"))
        
        print(f"Processing {len(mask_files)} images in '{diff}' category...")
        
        # 处理每个掩码文件
        success_count = 0
        for mask_file in tqdm(mask_files):
            # 获取文件名和对应的输入图像
            filename = get_filename(mask_file)
            input_file = None
            
            for ext in ['.png', '.jpg', '.jpeg']:
                input_path = os.path.join(input_subdir, f"{filename}{ext}")
                if os.path.exists(input_path):
                    input_file = input_path
                    break
            
            if input_file is None:
                print(f"Warning: Could not find corresponding input image for mask {mask_file}")
                continue
            
            # 处理图像
            output_path = os.path.join(output_subdir, f"{filename}.jpg")
            if process_image_with_mask(input_file, mask_file, output_path):
                success_count += 1
        
        print(f"Successfully processed {success_count} out of {len(mask_files)} images in '{diff}' category.")
    
    print("All processing completed!")


if __name__ == "__main__":
    main()