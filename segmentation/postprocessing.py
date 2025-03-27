#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版后处理模块，提供更强大的形态学操作、连通区域分析和边界平滑功能
"""

import cv2
import numpy as np


def apply_morphology(mask, operations=None):
    """
    应用形态学操作
    
    参数:
        mask: 输入掩码
        operations: 包含操作参数的字典
    
    返回:
        处理后的掩码
    """
    if operations is None:
        operations = {
            'opening': {'enabled': True, 'kernel_size': 3, 'iterations': 1},
            'closing': {'enabled': True, 'kernel_size': 5, 'iterations': 2},
            'dilation': {'enabled': True, 'kernel_size': 5, 'iterations': 2},
            'erosion': {'enabled': False, 'kernel_size': 3, 'iterations': 1}
        }
    
    result = mask.copy()
    
    # 形态学开操作（去除小噪点）
    if operations.get('opening', {}).get('enabled', False):
        kernel_size = operations.get('opening', {}).get('kernel_size', 3)
        iterations = operations.get('opening', {}).get('iterations', 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # 形态学闭操作（填补小孔洞）
    if operations.get('closing', {}).get('enabled', False):
        kernel_size = operations.get('closing', {}).get('kernel_size', 5)
        iterations = operations.get('closing', {}).get('iterations', 2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # 膨胀（扩展前景区域）
    if operations.get('dilation', {}).get('enabled', False):
        kernel_size = operations.get('dilation', {}).get('kernel_size', 5)
        iterations = operations.get('dilation', {}).get('iterations', 2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.dilate(result, kernel, iterations=iterations)
    
    # 腐蚀（缩小前景区域）
    if operations.get('erosion', {}).get('enabled', False):
        kernel_size = operations.get('erosion', {}).get('kernel_size', 3)
        iterations = operations.get('erosion', {}).get('iterations', 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.erode(result, kernel, iterations=iterations)
    
    return result


def analyze_connected_components(mask, min_area_ratio=0.001, max_num_components=5):
    """
    连通区域分析，增强版本能够更好地处理分离的区域
    
    参数:
        mask: 输入掩码
        min_area_ratio: 相对于图像大小的最小连通区域比例
        max_num_components: 保留的最大连通区域数量
        
    返回:
        处理后的掩码
    """
    # 连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 计算最小区域面积
    h, w = mask.shape
    min_area = min_area_ratio * h * w
    
    # 创建输出掩码
    result = np.zeros_like(mask)
    
    # 如果只有背景区域或没有足够大的区域
    if num_labels <= 1 or np.max([stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels) if i < len(stats)]) < min_area:
        # 返回原始掩码，不进行过滤
        return mask
    
    # 按面积降序排序连通区域
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if i < len(stats)]  # 跳过背景 (label 0)
    areas.sort(key=lambda x: x[1], reverse=True)
    
    # 保留足够大的连通区域（最多max_num_components个）
    regions_to_keep = []
    for i, (label, area) in enumerate(areas):
        if i >= max_num_components:
            break
        if area >= min_area:
            regions_to_keep.append(label)
    
    # 如果没有足够大的区域，至少保留最大的一个
    if not regions_to_keep and areas:
        regions_to_keep.append(areas[0][0])
    
    # 填充结果掩码
    for label in regions_to_keep:
        result[labels == label] = 255
    
    # 如果结果掩码为空，返回原始掩码
    if np.sum(result) == 0:
        return mask
    
    return result


def improved_fill_holes(mask, min_hole_area=5):
    """
    改进的孔洞填充算法，更好地处理复杂形状
    
    参数:
        mask: 输入掩码
        min_hole_area: 要填充的最小孔洞面积
        
    返回:
        填充孔洞后的掩码
    """
    # 确保输入掩码是二值图像
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 复制输入掩码
    filled = binary_mask.copy()
    
    # 方法1：使用形态学闭操作填充小孔洞
    kernel = np.ones((5, 5), np.uint8)
    morph_filled = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 方法2：使用洪水填充法处理较大的孔洞
    # 创建一个略大的图像，确保边界区域连通
    h, w = binary_mask.shape
    padded = np.zeros((h+2, w+2), np.uint8)
    padded[1:-1, 1:-1] = binary_mask
    
    # 用于洪水填充的掩码
    flood_mask = np.zeros((h+4, w+4), np.uint8)
    
    # 从边界开始洪水填充
    cv2.floodFill(padded, flood_mask, (0, 0), 255)
    
    # 取反，获取内部区域
    padded = cv2.bitwise_not(padded)
    
    # 裁剪回原始大小
    flood_filled = padded[1:-1, 1:-1]
    
    # 与原始掩码合并
    combined = cv2.bitwise_or(binary_mask, flood_filled)
    
    # 方法3：使用轮廓分析填充孔洞
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    
    contour_filled = binary_mask.copy()
    
    if contours and hierarchy is not None and len(hierarchy) > 0:
        for i, contour in enumerate(contours):
            # 确保层次结构数组不为空且索引有效
            if i < len(hierarchy[0]):
                # 仅填充内部轮廓（孔洞）
                if hierarchy[0][i][3] >= 0:  # 有父轮廓表示是孔洞
                    area = cv2.contourArea(contour)
                    if area >= min_hole_area:
                        cv2.drawContours(contour_filled, [contour], 0, 255, -1)
    
    # 组合所有方法的结果
    final_filled = cv2.bitwise_or(combined, contour_filled)
    
    # 应用一次中值滤波平滑结果
    final_filled = cv2.medianBlur(final_filled, 5)
    
    return final_filled


def improved_smooth_contours(mask, method='combined', kernel_size=5):
    """
    改进的边界平滑算法
    
    参数:
        mask: 输入掩码
        method: 平滑方法，可选 'gaussian', 'median', 'combined', 'contour'
        kernel_size: 核大小
        
    返回:
        平滑后的掩码
    """
    # 确保输入掩码是二值图像
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    if method == 'combined':
        # 结合高斯和中值滤波的效果
        # 先应用高斯滤波
        gaussian_smoothed = cv2.GaussianBlur(binary_mask, (kernel_size, kernel_size), 0)
        _, gaussian_smoothed = cv2.threshold(gaussian_smoothed, 127, 255, cv2.THRESH_BINARY)
        
        # 再应用中值滤波
        median_smoothed = cv2.medianBlur(gaussian_smoothed, kernel_size)
        
        # 最后再次二值化
        _, smoothed = cv2.threshold(median_smoothed, 127, 255, cv2.THRESH_BINARY)
        
    elif method == 'contour':
        # 使用轮廓重绘的方式平滑
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建空白画布
        smoothed = np.zeros_like(binary_mask)
        
        # 对每个轮廓应用平滑
        for contour in contours:
            # 对轮廓点进行平滑处理
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 绘制平滑后的轮廓并填充
            cv2.drawContours(smoothed, [approx], 0, 255, -1)
        
    elif method == 'gaussian':
        # 应用高斯滤波
        smoothed = cv2.GaussianBlur(binary_mask, (kernel_size, kernel_size), 0)
        _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
        
    elif method == 'median':
        # 应用中值滤波
        smoothed = cv2.medianBlur(binary_mask, kernel_size)
        
    else:
        # 默认不做处理
        smoothed = binary_mask.copy()
    
    return smoothed


def merge_segmentation_masks(masks, weights=None):
    """
    智能合并多个分割掩码，可以指定不同掩码的权重
    
    参数:
        masks: 掩码字典或列表
        weights: 权重字典，与masks的键对应
        
    返回:
        合并后的掩码
    """
    if not masks:
        return None
    
    # 如果是字典，获取掩码列表和对应权重
    if isinstance(masks, dict):
        if weights is None:
            # 默认权重，优先考虑颜色信息
            weights = {
                'color': 1.5,     # 颜色分割权重高
                'threshold': 0.7,
                'edge': 0.8,
                'region': 0.7,
                'yellow': 1.8     # 黄色检测权重最高
            }
        
        # 根据可用的掩码进行调整
        available_weights = {}
        for key in masks:
            if key in weights:
                available_weights[key] = weights[key]
            else:
                available_weights[key] = 0.5  # 默认权重
        
        # 创建加权掩码
        h, w = next(iter(masks.values())).shape
        weighted_sum = np.zeros((h, w), dtype=np.float32)
        total_weight = 0
        
        for key, mask in masks.items():
            if key in available_weights and key != 'final':  # 跳过最终掩码
                weight = available_weights[key]
                weighted_sum += (mask / 255.0) * weight
                total_weight += weight
        
        # 归一化并二值化
        if total_weight > 0:
            normalized = weighted_sum / total_weight
            threshold = 0.4  # 降低阈值，让更多区域被识别为前景
            merged = (normalized > threshold).astype(np.uint8) * 255
        else:
            # 如果没有有效权重，使用简单的逻辑或
            merged = np.zeros((h, w), dtype=np.uint8)
            for key, mask in masks.items():
                if key != 'final':  # 跳过最终掩码
                    merged = cv2.bitwise_or(merged, mask)
    
    else:  # 如果是列表
        if not masks:
            return None
        
        # 使用简单的逻辑或
        merged = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            if mask is not None:
                merged = cv2.bitwise_or(merged, mask)
    
    return merged


def postprocess_mask(masks, features, config=None):
    """
    对分割掩码进行增强后处理
    
    参数:
        masks: 分割方法生成的掩码集合
        features: 分割过程中提取的特征
        config: 后处理配置参数
        
    返回:
        后处理后的最终掩码
    """
    if config is None:
        config = {
            'morphology': {
                'enabled': True,
                'opening': {'enabled': True, 'kernel_size': 3, 'iterations': 1},
                'closing': {'enabled': True, 'kernel_size': 7, 'iterations': 3},
                'dilation': {'enabled': True, 'kernel_size': 5, 'iterations': 2},
                'erosion': {'enabled': False, 'kernel_size': 3, 'iterations': 1}
            },
            'connected_components': {
                'enabled': True,
                'min_area_ratio': 0.001,  # 减小这个值，保留更小的区域
                'max_num_components': 5   # 增加这个值，保留更多的区域
            },
            'contour_smoothing': {
                'enabled': True,
                'method': 'combined',  # 使用组合平滑方法
                'kernel_size': 5
            },
            'hole_filling': {
                'enabled': True,
                'min_hole_area': 5  # 减小这个值，填充更小的孔洞
            }
        }
    
    # 检查掩码是否可用
    if not masks:
        return None
    
    # 智能合并所有掩码创建初始掩码 - 使用改进的合并算法
    initial_mask = merge_segmentation_masks(masks)
    
    if initial_mask is None:
        return None
    
    # 如果初始掩码几乎为空，可能是合并阈值太高，使用颜色掩码
    if np.sum(initial_mask) < 100 and 'color' in masks:
        initial_mask = masks['color']
    elif np.sum(initial_mask) < 100 and 'yellow' in masks:
        initial_mask = masks['yellow']
    
    # 保存中间结果
    result = initial_mask.copy()
    
    # 对于几乎为空的掩码，进行更多的形态学膨胀
    if np.sum(result) < 1000:
        kernel = np.ones((7, 7), np.uint8)
        result = cv2.dilate(result, kernel, iterations=3)
    
    # 填充孔洞 - 使用改进版本
    if config.get('hole_filling', {}).get('enabled', False):
        result = improved_fill_holes(
            result,
            min_hole_area=config.get('hole_filling', {}).get('min_hole_area', 5)
        )
    
    # 形态学操作
    if config.get('morphology', {}).get('enabled', False):
        result = apply_morphology(result, config.get('morphology'))
    
    # 连通区域分析
    if config.get('connected_components', {}).get('enabled', False):
        result = analyze_connected_components(
            result,
            min_area_ratio=config.get('connected_components', {}).get('min_area_ratio', 0.001),
            max_num_components=config.get('connected_components', {}).get('max_num_components', 5)
        )
    
    # 如果连通区域分析后掩码面积大幅减小，可能是过滤条件太严格，使用原始掩码
    if np.sum(result) < np.sum(initial_mask) * 0.3:
        result = initial_mask.copy()
        
        # 再次尝试形态学操作改善掩码质量
        if config.get('morphology', {}).get('enabled', False):
            result = apply_morphology(result, config.get('morphology'))
    
    # 边界平滑 - 使用改进版本
    if config.get('contour_smoothing', {}).get('enabled', False):
        result = improved_smooth_contours(
            result,
            method=config.get('contour_smoothing', {}).get('method', 'combined'),
            kernel_size=config.get('contour_smoothing', {}).get('kernel_size', 5)
        )
    
    # 最后一次形态学闭操作，确保结果平滑
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 最终确保结果不为空，如果为空则回退到初始掩码
    if np.sum(result) == 0:
        result = initial_mask
    
    return result