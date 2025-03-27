#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像处理工具模块，提供通用的图像处理功能
"""

import cv2
import numpy as np


def resize_keep_aspect(image, target_size):
    """
    保持宽高比例的图像缩放
    
    参数:
        image: 输入图像
        target_size: 目标尺寸 (width, height)
        
    返回:
        缩放后的图像
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    
    # 缩放图像
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 创建目标大小的黑色画布
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # 计算居中位置
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # 将图像放到画布中心
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def normalize_image(image, min_val=0, max_val=255):
    """
    图像归一化
    
    参数:
        image: 输入图像
        min_val: 归一化后的最小值
        max_val: 归一化后的最大值
        
    返回:
        归一化后的图像
    """
    # 计算当前的最小值和最大值
    img_min = np.min(image)
    img_max = np.max(image)
    
    # 避免除以零
    if img_max == img_min:
        return np.ones_like(image) * min_val
    
    # 执行归一化
    normalized = (image - img_min) / (img_max - img_min) * (max_val - min_val) + min_val
    
    # 转换为正确的数据类型
    if max_val <= 1.0:
        return normalized.astype(np.float32)
    else:
        return normalized.astype(np.uint8)


def apply_brightness_contrast(image, brightness=0, contrast=0):
    """
    调整图像的亮度和对比度
    
    参数:
        image: 输入图像
        brightness: 亮度调整值 (-100 to 100)
        contrast: 对比度调整值 (-100 to 100)
        
    返回:
        调整后的图像
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        # 应用亮度调整
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        
        # 应用对比度调整
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    return image


def auto_canny(image, sigma=0.33):
    """
    自动Canny边缘检测
    
    参数:
        image: 输入图像
        sigma: 阈值比例因子
        
    返回:
        边缘图像
    """
    # 确保图像是灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 计算中值
    v = np.median(gray)
    
    # 根据中值自动确定低阈值和高阈值
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    # 应用Canny边缘检测
    edges = cv2.Canny(gray, lower, upper)
    
    return edges


def convert_to_binary(image, method='otsu', threshold=127):
    """
    将图像转换为二值图像
    
    参数:
        image: 输入图像
        method: 二值化方法，可选 'otsu', 'adaptive', 'global'
        threshold: 全局阈值方法的阈值
        
    返回:
        二值图像
    """
    # 确保图像是灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == 'otsu':
        # Otsu自动阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        # 自适应阈值
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    else:
        # 全局阈值
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    return binary


def extract_contours(image, method='binary', min_area=100):
    """
    提取图像中的轮廓
    
    参数:
        image: 输入图像
        method: 提取方法，可选 'binary', 'canny'
        min_area: 最小轮廓面积
        
    返回:
        轮廓列表
    """
    # 确保图像是灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 获取二值图像
    if method == 'binary':
        binary = convert_to_binary(gray)
    else:  # 'canny'
        binary = auto_canny(gray)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 按面积过滤轮廓
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    
    return filtered_contours


def apply_color_filter(image, color, tolerance=30):
    """
    应用颜色过滤
    
    参数:
        image: 输入图像 (BGR)
        color: 目标颜色 (BGR)
        tolerance: 颜色容差
        
    返回:
        过滤后的二值掩码
    """
    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 将目标颜色转换为HSV
    target_color = np.uint8([[color]])
    target_hsv = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)
    target_hue = target_hsv[0, 0, 0]
    
    # 设置HSV阈值范围
    lower_bound = np.array([max(0, target_hue - tolerance), 50, 50])
    upper_bound = np.array([min(179, target_hue + tolerance), 255, 255])
    
    # 创建掩码
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    return mask


def remove_small_objects(binary_image, min_size=50):
    """
    移除二值图像中的小对象
    
    参数:
        binary_image: 二值图像
        min_size: 最小对象面积
        
    返回:
        处理后的二值图像
    """
    # 连通区域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # 创建输出图像
    result = np.zeros_like(binary_image)
    
    # 保留大于最小尺寸的区域
    for i in range(1, num_labels):  # 从1开始跳过背景
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            result[labels == i] = 255
    
    return result


def remove_background(image, mask):
    """
    根据掩码移除背景
    
    参数:
        image: 输入图像
        mask: 二值掩码
        
    返回:
        去除背景后的图像
    """
    # 确保二值掩码
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 应用掩码
    result = cv2.bitwise_and(image, image, mask=binary_mask)
    
    return result


def crop_to_roi(image, mask):
    """
    裁剪图像到感兴趣区域
    
    参数:
        image: 输入图像
        mask: 二值掩码
        
    返回:
        裁剪后的图像
    """
    # 查找掩码的非零像素
    y_indices, x_indices = np.where(mask > 0)
    
    # 如果掩码为空，返回原图
    if len(y_indices) == 0 or len(x_indices) == 0:
        return image
    
    # 计算边界框
    x_min, y_min = np.min(x_indices), np.min(y_indices)
    x_max, y_max = np.max(x_indices), np.max(y_indices)
    
    # 裁剪图像
    cropped = image[y_min:y_max+1, x_min:x_max+1]
    
    return cropped