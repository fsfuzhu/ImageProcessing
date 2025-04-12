#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - 通用版本
颜色空间转换工具
基于讲座1B - 数字图像和点处理
"""

import cv2
import numpy as np

def convert_to_hsv(image):
    """
    将RGB图像转换为HSV颜色空间
    
    HSV将颜色（色调）与强度（值）分离，使其对光照变化不太敏感，
    这对分割很有利。
    
    Args:
        image: RGB图像作为NumPy数组
        
    Returns:
        HSV图像作为NumPy数组
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def convert_to_lab(image):
    """
    将RGB图像转换为LAB颜色空间
    
    LAB将亮度（L）与颜色信息（A和B通道）分离，
    使其在区分颜色差异时不受亮度影响。
    
    Args:
        image: RGB图像作为NumPy数组
        
    Returns:
        LAB图像作为NumPy数组
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

def convert_to_grayscale(image):
    """
    使用加权公式将RGB图像转换为灰度
    
    如讲座1B中所述，我们可以使用以下方法将RGB转换为灰度：
    I = 0.30R + 0.59G + 0.11B（权重基于人眼敏感度）
    
    Args:
        image: RGB图像作为NumPy数组
        
    Returns:
        灰度图像作为NumPy数组
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def extract_hsv_channels(hsv_image):
    """
    从HSV图像中提取单独的通道
    
    Args:
        hsv_image: HSV图像作为NumPy数组
        
    Returns:
        (H, S, V)通道作为单独的NumPy数组的元组
    """
    h, s, v = cv2.split(hsv_image)
    return h, s, v

def extract_lab_channels(lab_image):
    """
    从LAB图像中提取单独的通道
    
    Args:
        lab_image: LAB图像作为NumPy数组
        
    Returns:
        (L, A, B)通道作为单独的NumPy数组的元组
    """
    l, a, b = cv2.split(lab_image)
    return l, a, b

def calculate_greenness(image):
    """
    计算RGB图像的"绿度"
    
    如讲座1B中所述，简单的绿度度量为：
    G - (R+B)/2
    
    Args:
        image: RGB图像作为NumPy数组
        
    Returns:
        绿度图像作为NumPy数组
    """
    b, g, r = cv2.split(image)
    greenness = g.astype(np.float32) - (r.astype(np.float32) + b.astype(np.float32)) / 2
    
    # 归一化到0-255范围
    greenness_normalized = np.clip(greenness, 0, None)
    if np.max(greenness_normalized) > 0:
        greenness_normalized = (greenness_normalized / np.max(greenness_normalized) * 255).astype(np.uint8)
    else:
        greenness_normalized = np.zeros_like(greenness, dtype=np.uint8)
    
    return greenness_normalized