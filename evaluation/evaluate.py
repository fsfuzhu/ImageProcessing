#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - 花朵分割评估模块
使用ground truth中的红色遮罩评估分割结果
"""

import os
import cv2
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path

def extract_red_mask(image_path):
    """
    从ground truth图像中提取红色遮罩
    
    Args:
        image_path: ground truth图像路径
        
    Returns:
        红色区域的二值遮罩 (255表示红色区域, 0表示背景)
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return None
    
    # 转换到HSV颜色空间, 更容易分离颜色
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义HSV空间中红色的范围
    # 红色在HSV中有两个范围 (0-10和170-180)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # 创建两个红色范围的掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # 合并掩码
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    return red_mask

def apply_mask_to_original(original_image_path, mask):
    """
    将mask应用到原始图像上，保留花朵，背景变黑
    
    Args:
        original_image_path: 原始图像路径
        mask: 二值掩码
        
    Returns:
        处理后的图像，花朵前景保留，背景变黑
    """
    # 读取原始图像
    original = cv2.imread(original_image_path)
    if original is None:
        print(f"错误: 无法读取原始图像 {original_image_path}")
        return None
    
    # 确保掩码是二值的
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # 如果原始图像和掩码大小不同，调整掩码大小
    if original.shape[:2] != binary_mask.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (original.shape[1], original.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
    
    # 创建3通道掩码
    mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    # 应用掩码到原始图像
    result = cv2.bitwise_and(original, mask_3ch)
    
    return result

def calculate_iou(predicted_mask, ground_truth_mask):
    """
    计算IoU (Intersection over Union)
    
    Args:
        predicted_mask: 预测的二值掩码
        ground_truth_mask: ground truth二值掩码
        
    Returns:
        IoU值 (0-1之间的浮点数)
    """
    # 确保掩码是二值的
    _, pred_binary = cv2.threshold(predicted_mask, 1, 1, cv2.THRESH_BINARY)
    _, gt_binary = cv2.threshold(ground_truth_mask, 1, 1, cv2.THRESH_BINARY)
    
    # 计算交集和并集
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # 避免除零错误
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_dice_coefficient(predicted_mask, ground_truth_mask):
    """
    计算Dice系数 (F1分数)
    
    Args:
        predicted_mask: 预测的二值掩码
        ground_truth_mask: ground truth二值掩码
        
    Returns:
        Dice系数 (0-1之间的浮点数)
    """
    # 确保掩码是二值的
    _, pred_binary = cv2.threshold(predicted_mask, 1, 1, cv2.THRESH_BINARY)
    _, gt_binary = cv2.threshold(ground_truth_mask, 1, 1, cv2.THRESH_BINARY)
    
    # 计算交集
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    
    # 计算两个掩码的大小
    size_pred = pred_binary.sum()
    size_gt = gt_binary.sum()
    
    # 避免除零错误
    if size_pred + size_gt == 0:
        return 0.0
    
    # 计算Dice系数
    return 2 * intersection / (size_pred + size_gt)

def evaluate_segmentation_with_red_mask(input_dir, segmented_dir, ground_truth_dir, difficulty_levels=None, output_dir=None):
    """
    使用ground truth中的红色遮罩评估分割结果
    
    Args:
        input_dir: 原始输入图像目录
        segmented_dir: 分割结果目录
        ground_truth_dir: ground truth目录
        difficulty_levels: 难度级别列表 (例如 ['easy', 'medium', 'hard'])
        output_dir: 输出评估结果的目录
        
    Returns:
        评估指标的字典
    """
    # 如果没有指定难度级别，默认处理所有图像
    if difficulty_levels is None:
        difficulty_levels = [""]
    
    # 确保输出目录存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 初始化结果列表和累计指标
    all_results = []
    total_iou = 0
    total_accuracy = 0
    total_dice = 0
    total_images = 0
    
    print("评估分割结果:")
    
    # 处理每个难度级别
    for level in difficulty_levels:
        # 构建对应难度级别的目录路径
        level_input_dir = os.path.join(input_dir, level) if level else input_dir
        level_segmented_dir = os.path.join(segmented_dir, level) if level else segmented_dir
        level_gt_dir = os.path.join(ground_truth_dir, level) if level else ground_truth_dir
        
        if output_dir:
            level_output_dir = os.path.join(output_dir, level) if level else output_dir
            os.makedirs(level_output_dir, exist_ok=True)
        
        # 查找所有分割结果
        segmented_paths = glob.glob(os.path.join(level_segmented_dir, "*.jpg")) + \
                          glob.glob(os.path.join(level_segmented_dir, "*.png"))
        
        print(f"处理 {len(segmented_paths)} 张图像，难度级别: {level or '默认'}")
        
        # 处理每张图像
        for seg_path in segmented_paths:
            # 获取图像文件名
            img_filename = os.path.basename(seg_path)
            base_name = os.path.splitext(img_filename)[0]
            
            # 查找对应的ground truth文件
            possible_gt_paths = [
                os.path.join(level_gt_dir, f"{base_name}.png"),
                os.path.join(level_gt_dir, f"{base_name}.jpg")
            ]
            
            gt_path = None
            for path in possible_gt_paths:
                if os.path.exists(path):
                    gt_path = path
                    break
            
            # 跳过找不到ground truth的图像
            if gt_path is None:
                print(f"  警告: 找不到 {img_filename} 的ground truth")
                continue
                
            # 查找对应的原始输入图像
            possible_input_paths = [
                os.path.join(level_input_dir, f"{base_name}.png"),
                os.path.join(level_input_dir, f"{base_name}.jpg")
            ]
            
            input_path = None
            for path in possible_input_paths:
                if os.path.exists(path):
                    input_path = path
                    break
            
            # 跳过找不到原始图像的情况
            if input_path is None:
                print(f"  警告: 找不到 {img_filename} 的原始输入图像")
                continue
            
            print(f"  处理 {img_filename}...")
            
            try:
                # 1. 从ground truth中提取红色遮罩
                red_mask = extract_red_mask(gt_path)
                if red_mask is None:
                    continue
                
                # 2. 读取分割结果
                segmented = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                if segmented is None:
                    print(f"  错误: 无法读取分割结果 {seg_path}")
                    continue
                
                # 3. 将红色遮罩应用到原始图像上，获取ground truth花朵
                ground_truth_flower = apply_mask_to_original(input_path, red_mask)
                if ground_truth_flower is None:
                    continue
                
                # 4. 确保分割结果和ground truth大小相同
                if segmented.shape != red_mask.shape:
                    segmented = cv2.resize(segmented, (red_mask.shape[1], red_mask.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                
                # 5. 将分割结果二值化
                _, binary_segmented = cv2.threshold(segmented, 1, 255, cv2.THRESH_BINARY)
                
                # 6. 计算评估指标
                iou = calculate_iou(binary_segmented, red_mask)
                
                # 计算准确率
                binary_segmented_flat = (binary_segmented > 0).flatten()
                red_mask_flat = (red_mask > 0).flatten()
                accuracy = accuracy_score(red_mask_flat, binary_segmented_flat)
                
                # 计算Dice系数
                dice = calculate_dice_coefficient(binary_segmented, red_mask)
                
                # 7. 添加到结果列表
                result = {
                    'image': img_filename,
                    'difficulty': level or 'default',
                    'iou': iou,
                    'accuracy': accuracy,
                    'dice': dice
                }
                all_results.append(result)
                
                # 8. 累计指标
                total_iou += iou
                total_accuracy += accuracy
                total_dice += dice
                total_images += 1
                
                print(f"  {img_filename}: IoU = {iou:.4f}, Accuracy = {accuracy:.4f}, Dice = {dice:.4f}")
                
                # 9. 如果指定了输出目录，保存可视化结果
                if output_dir:
                    # 创建可视化比较
                    vis_path = os.path.join(level_output_dir, f"{base_name}_comparison.png")
                    
                    # 原始图像
                    original = cv2.imread(input_path)
                    
                    # 分割结果应用到原始图像
                    segmented_color = cv2.imread(seg_path)
                    
                    # 创建可视化图像
                    # 确保所有图像具有相同的大小
                    h, w = original.shape[:2]
                    segmented_resized = cv2.resize(segmented_color, (w, h))
                    ground_truth_resized = cv2.resize(ground_truth_flower, (w, h))
                    
                    # 水平拼接原始图像、分割结果和ground truth
                    comparison = np.hstack((original, segmented_resized, ground_truth_resized))
                    
                    # 添加文本标签
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    y_pos = 30
                    cv2.putText(comparison, "Original", (10, y_pos), font, 1, (255, 255, 255), 2)
                    cv2.putText(comparison, "Segmented", (w + 10, y_pos), font, 1, (255, 255, 255), 2)
                    cv2.putText(comparison, "Ground Truth", (2*w + 10, y_pos), font, 1, (255, 255, 255), 2)
                    cv2.putText(comparison, f"IoU: {iou:.4f}", (10, h - 10), font, 0.7, (255, 255, 255), 2)
                    
                    # 保存比较图像
                    cv2.imwrite(vis_path, comparison)
                
            except Exception as e:
                print(f"  处理 {img_filename} 时出错: {e}")
    
    # 计算平均指标
    avg_metrics = {}
    if total_images > 0:
        avg_metrics['avg_iou'] = total_iou / total_images
        avg_metrics['avg_accuracy'] = total_accuracy / total_images
        avg_metrics['avg_dice'] = total_dice / total_images
        avg_metrics['total_images'] = total_images
        
        print(f"\n评估完成，处理了 {total_images} 张图像")
        print(f"平均 IoU: {avg_metrics['avg_iou']:.4f}")
        print(f"平均准确率: {avg_metrics['avg_accuracy']:.4f}")
        print(f"平均 Dice 系数: {avg_metrics['avg_dice']:.4f}")
    
    # 如果指定了输出目录，保存评估结果
    if output_dir and all_results:
        # 保存为CSV
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, "segmentation_evaluation.csv")
        results_df.to_csv(csv_path, index=False)
        
        # 保存总结报告
        summary_path = os.path.join(output_dir, "evaluation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("花朵分割评估结果\n")
            f.write("====================\n\n")
            f.write(f"评估的图像数量: {total_images}\n\n")
            f.write("平均指标:\n")
            f.write(f"- IoU: {avg_metrics['avg_iou']:.4f}\n")
            f.write(f"- 准确率: {avg_metrics['avg_accuracy']:.4f}\n")
            f.write(f"- Dice系数: {avg_metrics['avg_dice']:.4f}\n\n")
            
            # 按难度级别分组
            if len(difficulty_levels) > 1:
                f.write("按难度级别的平均指标:\n")
                for level in difficulty_levels:
                    level_results = [r for r in all_results if r['difficulty'] == (level or 'default')]
                    if level_results:
                        level_iou = sum(r['iou'] for r in level_results) / len(level_results)
                        level_acc = sum(r['accuracy'] for r in level_results) / len(level_results)
                        level_dice = sum(r['dice'] for r in level_results) / len(level_results)
                        
                        f.write(f"\n{level or '默认'} 级别 ({len(level_results)}张图像):\n")
                        f.write(f"- IoU: {level_iou:.4f}\n")
                        f.write(f"- 准确率: {level_acc:.4f}\n")
                        f.write(f"- Dice系数: {level_dice:.4f}\n")
        
        print(f"评估结果已保存到 {output_dir}")
    
    return avg_metrics