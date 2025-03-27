#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
后处理模块，提供形态学操作、连通区域分析、边界平滑等功能
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
            'closing': {'enabled': True, 'kernel_size': 3, 'iterations': 2},
            'dilation': {'enabled': True, 'kernel_size': 3, 'iterations': 1},
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
        kernel_size = operations.get('closing', {}).get('kernel_size', 3)
        iterations = operations.get('closing', {}).get('iterations', 2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # 膨胀（扩展前景区域）
    if operations.get('dilation', {}).get('enabled', False):
        kernel_size = operations.get('dilation', {}).get('kernel_size', 3)
        iterations = operations.get('dilation', {}).get('iterations', 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.dilate(result, kernel, iterations=iterations)
    
    # 腐蚀（缩小前景区域）
    if operations.get('erosion', {}).get('enabled', False):
        kernel_size = operations.get('erosion', {}).get('kernel_size', 3)
        iterations = operations.get('erosion', {}).get('iterations', 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.erode(result, kernel, iterations=iterations)
    
    return result


def analyze_connected_components(mask, min_area_ratio=0.01, max_num_components=3):
    """
    连通区域分析，去除小区域，保留最大的几个连通区域
    
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
    
    # 按面积降序排序连通区域
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]  # 跳过背景 (label 0)
    areas.sort(key=lambda x: x[1], reverse=True)
    
    # 保留最大的几个连通区域
    for i, (label, area) in enumerate(areas):
        if i >= max_num_components or area < min_area:
            break
        
        # 将该连通区域添加到结果中
        result[labels == label] = 255
    
    return result


def smooth_contours(mask, method='gaussian', kernel_size=5):
    """
    平滑分割边界
    
    参数:
        mask: 输入掩码
        method: 平滑方法，可选 'gaussian', 'median'
        kernel_size: 核大小
        
    返回:
        平滑后的掩码
    """
    # 应用平滑处理
    if method == 'gaussian':
        smoothed = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    elif method == 'median':
        smoothed = cv2.medianBlur(mask, kernel_size)
    else:
        smoothed = mask.copy()
    
    # 再次二值化，确保结果仍是二值图像
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    
    return smoothed


def fill_holes(mask, min_hole_area=50):
    """
    填充掩码中的孔洞
    
    参数:
        mask: 输入掩码
        min_hole_area: 要填充的最小孔洞面积
        
    返回:
        填充孔洞后的掩码
    """
    # 复制输入掩码
    filled = mask.copy()
    
    # 找到轮廓
    contours, hierarchy = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # 遍历轮廓
    for i, contour in enumerate(contours):
        # 如果是孔洞（内部轮廓）
        if hierarchy[0][i][3] >= 0:  # 有父轮廓
            # 计算孔洞面积
            area = cv2.contourArea(contour)
            
            # 如果面积大于阈值，填充该孔洞
            if area > min_hole_area:
                cv2.drawContours(filled, [contour], 0, 255, -1)
    
    return filled


def postprocess_mask(masks, features, config=None):
    """
    对分割掩码进行后处理
    
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
                'closing': {'enabled': True, 'kernel_size': 3, 'iterations': 2},
                'dilation': {'enabled': True, 'kernel_size': 3, 'iterations': 1},
                'erosion': {'enabled': False, 'kernel_size': 3, 'iterations': 1}
            },
            'connected_components': {
                'enabled': True,
                'min_area_ratio': 0.01,
                'max_num_components': 3
            },
            'contour_smoothing': {
                'enabled': True,
                'method': 'gaussian',
                'kernel_size': 5
            },
            'hole_filling': {
                'enabled': True,
                'min_hole_area': 50
            }
        }
    
    # 获取最终掩码
    if 'final' in masks:
        result = masks['final']
    else:
        # 如果没有最终掩码，使用颜色掩码
        result = masks.get('color', np.zeros_like(next(iter(masks.values()))))
    
    # 形态学操作
    if config.get('morphology', {}).get('enabled', False):
        result = apply_morphology(result, config.get('morphology'))
    
    # 连通区域分析
    if config.get('connected_components', {}).get('enabled', False):
        result = analyze_connected_components(
            result,
            min_area_ratio=config.get('connected_components', {}).get('min_area_ratio', 0.01),
            max_num_components=config.get('connected_components', {}).get('max_num_components', 3)
        )
    
    # 填充孔洞
    if config.get('hole_filling', {}).get('enabled', False):
        result = fill_holes(
            result,
            min_hole_area=config.get('hole_filling', {}).get('min_hole_area', 50)
        )
    
    # 边界平滑
    if config.get('contour_smoothing', {}).get('enabled', False):
        result = smooth_contours(
            result,
            method=config.get('contour_smoothing', {}).get('method', 'gaussian'),
            kernel_size=config.get('contour_smoothing', {}).get('kernel_size', 5)
        )
    
    return result