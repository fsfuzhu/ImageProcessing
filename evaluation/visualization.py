#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化模块，提供分割结果和评估指标的可视化功能
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    将掩码覆盖到原始图像上
    
    参数:
        image: 原始图像
        mask: 二值掩码
        color: 掩码颜色 (B, G, R)
        alpha: 透明度
        
    返回:
        叠加了掩码的图像
    """
    # 确保二值掩码
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 创建彩色掩码
    colored_mask = np.zeros_like(image)
    colored_mask[binary_mask > 0] = color
    
    # 叠加到原始图像
    overlayed = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    return overlayed


def visualize_segmentation_comparison(image, mask, ground_truth, save_path=None):
    """
    可视化分割结果与真实掩码的比较
    
    参数:
        image: 原始图像
        mask: 预测的掩码
        ground_truth: 真实掩码
        save_path: 保存路径，如果不为None则保存图像
        
    返回:
        可视化结果图像
    """
    # 确保二值掩码
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, gt_binary = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
    
    # 创建可视化图像
    comparison = np.zeros_like(image)
    
    # 正确分割的区域 (真阳性) - 绿色
    true_positive = np.logical_and(mask_binary > 0, gt_binary > 0)
    comparison[true_positive] = (0, 255, 0)  # 绿色
    
    # 错误分割的区域 (假阳性) - 红色
    false_positive = np.logical_and(mask_binary > 0, gt_binary == 0)
    comparison[false_positive] = (0, 0, 255)  # 红色
    
    # 漏分割的区域 (假阴性) - 蓝色
    false_negative = np.logical_and(mask_binary == 0, gt_binary > 0)
    comparison[false_negative] = (255, 0, 0)  # 蓝色
    
    # 叠加到原始图像
    result = cv2.addWeighted(image, 0.7, comparison, 0.3, 0)
    
    # 添加边框区分不同区域
    h, w = image.shape[:2]
    vis_image = np.ones((h, w*3, 3), dtype=np.uint8) * 255
    
    # 原始图像
    vis_image[:, :w] = image
    
    # 预测掩码
    mask_overlay = overlay_mask(image, mask_binary)
    vis_image[:, w:2*w] = mask_overlay
    
    # 比较结果
    vis_image[:, 2*w:] = result
    
    # 添加文本标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_image, "Original", (10, 30), font, 1, (0, 0, 0), 2)
    cv2.putText(vis_image, "Prediction", (w+10, 30), font, 1, (0, 0, 0), 2)
    cv2.putText(vis_image, "Comparison", (2*w+10, 30), font, 1, (0, 0, 0), 2)
    
    # 保存结果
    if save_path:
        cv2.imwrite(save_path, vis_image)
    
    return vis_image


def visualize_pipeline_steps(steps, save_dir=None):
    """
    可视化处理流程的各个步骤
    
    参数:
        steps: 包含流程各步骤图像的字典
        save_dir: 保存目录，如果不为None则保存图像
        
    返回:
        可视化结果图像
    """
    # 计算画布大小
    num_steps = len(steps)
    cols = min(4, num_steps)  # 最多4列
    rows = (num_steps + cols - 1) // cols  # 计算所需行数
    
    # 假设所有图像具有相同的尺寸
    sample_img = list(steps.values())[0]
    h, w = sample_img.shape[:2]
    
    # 创建画布
    canvas = np.ones((h*rows, w*cols, 3), dtype=np.uint8) * 255
    
    # 填充画布
    for i, (name, img) in enumerate(steps.items()):
        row = i // cols
        col = i % cols
        
        # 确保图像是3通道
        if len(img.shape) == 2 or img.shape[2] == 1:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()
        
        # 将图像放到画布上
        canvas[row*h:(row+1)*h, col*w:(col+1)*w] = img_color
        
        # 添加步骤名称
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, name, (col*w+10, row*h+30), font, 0.7, (0, 0, 0), 2)
    
    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "pipeline_steps.jpg"), canvas)
    
    return canvas


def plot_evaluation_metrics(metrics_list, names=None, save_path=None):
    """
    绘制评估指标的柱状图
    
    参数:
        metrics_list: 评估指标列表
        names: 各组指标的名称
        save_path: 保存路径，如果不为None则保存图像
    """
    if not metrics_list:
        return
    
    # 提取所有指标名称
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    all_metrics = sorted(list(all_metrics))
    
    # 准备数据
    n_groups = len(metrics_list)
    if names is None:
        names = [f'Group {i+1}' for i in range(n_groups)]
    
    # 设置图表
    fig, ax = plt.subplots(figsize=(12, 6))
    index = np.arange(len(all_metrics))
    bar_width = 0.8 / n_groups
    opacity = 0.8
    
    # 绘制柱状图
    for i, (metrics, name) in enumerate(zip(metrics_list, names)):
        values = [metrics.get(metric, 0) for metric in all_metrics]
        offset = bar_width * i - bar_width * n_groups / 2 + bar_width / 2
        plt.bar(index + offset, values, bar_width,
                alpha=opacity, label=name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Evaluation Metrics Comparison')
    plt.xticks(index, all_metrics, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_heatmap(image, mask, ground_truth=None, save_path=None):
    """
    创建热力图展示分割差异
    
    参数:
        image: 原始图像
        mask: 预测的掩码
        ground_truth: 真实掩码（可选）
        save_path: 保存路径，如果不为None则保存图像
        
    返回:
        热力图图像
    """
    # 确保二值掩码
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 创建热力图
    heatmap = np.zeros_like(image)
    
    if ground_truth is not None:
        # 确保二值GT掩码
        _, gt_binary = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
        
        # 计算差异
        diff = np.abs(mask_binary.astype(np.float32) - gt_binary.astype(np.float32))
        
        # 创建自定义颜色映射: 蓝色(低差异) -> 红色(高差异)
        cmap = LinearSegmentedColormap.from_list('diff_cmap', [(0, 0, 1), (1, 0, 0)])
        
        # 转换为热力图
        heatmap_data = diff / 255.0
        heatmap_img = plt.cm.get_cmap(cmap)(heatmap_data)
        heatmap_img = (heatmap_img[:, :, :3] * 255).astype(np.uint8)
        
        # 叠加到原始图像
        heatmap = cv2.addWeighted(image, 0.7, heatmap_img, 0.3, 0)
    else:
        # 如果没有ground truth，仅显示掩码
        mask_color = cv2.applyColorMap(mask_binary, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)
    
    # 保存结果
    if save_path:
        cv2.imwrite(save_path, heatmap)
    
    return heatmap


def save_visualization(output_dir, images_dict):
    """
    保存可视化结果
    
    参数:
        output_dir: 输出目录
        images_dict: 包含待保存图像的字典
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存各个图像
    for name, image in images_dict.items():
        # 确保图像是3通道
        if image is not None:
            if len(image.shape) == 2 or image.shape[2] == 1:
                image_to_save = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_to_save = image.copy()
            
            # 保存图像
            save_path = os.path.join(output_dir, f"{name}.jpg")
            cv2.imwrite(save_path, image_to_save)
    
    # 如果有完整的处理流程，创建流程图
    if len(images_dict) > 2:
        pipeline_path = os.path.join(output_dir, "pipeline_visualization.jpg")
        visualize_pipeline_steps(images_dict, save_path=pipeline_path)