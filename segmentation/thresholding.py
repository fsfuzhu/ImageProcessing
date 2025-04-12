#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - 通用版本
阈值处理实现
基于讲座4 - 阈值处理和二值图像
"""

import cv2
import numpy as np
from skimage import filters

def simple_threshold(image, threshold=127):
    """
    应用简单的全局阈值
    
    如讲座4中所述，这是最基本的阈值处理方法，
    其中高于固定阈值的像素被视为前景。
    
    Args:
        image: 灰度输入图像
        threshold: 阈值（0-255）
        
    Returns:
        二值图像
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def otsu_threshold(image):
    """
    应用Otsu的自动阈值处理
    
    Otsu的方法通过最小化前景和背景之间的类内方差，
    自动确定最佳阈值。
    
    Args:
        image: 灰度输入图像
        
    Returns:
        二值图像
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def adaptive_threshold(image, block_size=15, c=3):
    """
    应用自适应阈值处理
    
    自适应阈值处理计算图像不同区域的不同阈值，
    使其对光照变化更为稳健。
    
    Args:
        image: 灰度输入图像
        block_size: 像素邻域的大小（必须是奇数）
        c: 从均值中减去的常数
        
    Returns:
        二值图像
    """
    # 确保图像是8位灰度图
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # 确保block_size为奇数且至少为3
    block_size = max(3, block_size)
    if block_size % 2 == 0:
        block_size += 1
        
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c
    )

def rosin_threshold(image):
    """
    应用Rosin的单峰阈值处理
    
    如讲座4中所述，Rosin的方法对于没有明显双峰分布的单峰直方图很有效。
    
    Args:
        image: 灰度输入图像
        
    Returns:
        二值图像
    """
    # 计算直方图
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    # 找到直方图的峰值
    peak_idx = np.argmax(hist)
    
    # 找到最后一个非零bin
    last_idx = 255
    while hist[last_idx] == 0 and last_idx > peak_idx:
        last_idx -= 1
    
    # 计算从峰值到最后bin的线
    x1, y1 = peak_idx, hist[peak_idx]
    x2, y2 = last_idx, hist[last_idx]
    
    # 计算每个点到线的垂直距离
    max_dist = 0
    threshold = peak_idx
    
    for i in range(peak_idx + 1, last_idx):
        if hist[i] == 0:
            continue
        
        # 计算垂直距离
        # 线方程：ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        dist = abs(a * i + b * hist[i] + c) / np.sqrt(a**2 + b**2)
        
        if dist > max_dist:
            max_dist = dist
            threshold = i
    
    # 应用计算出的阈值
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def multi_level_threshold(image, num_classes=3):
    """
    应用多级阈值处理，用于更复杂的图像
    
    这超出了简单的二值阈值处理，可以分离多个类
    （例如，背景、花瓣、花的中心）。
    
    Args:
        image: 灰度输入图像
        num_classes: 要分离的类别数
        
    Returns:
        多级阈值处理后的图像
    """
    try:
        # 使用scikit-image的多级Otsu实现
        thresholds = filters.threshold_multiotsu(image, classes=num_classes)
        
        # 创建结果图像
        result = np.digitize(image, bins=thresholds)
        
        # 缩放结果到0-255范围
        result = (255 * result / (num_classes - 1)).astype(np.uint8)
        
        return result
    except Exception as e:
        print(f"多级阈值处理失败: {e}")
        # 回退到Otsu
        return otsu_threshold(image)