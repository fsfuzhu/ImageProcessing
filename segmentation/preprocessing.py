#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像预处理模块，提供尺寸调整、去噪、对比度增强等功能
"""

import cv2
import numpy as np


def resize_image(image, width, height):
    """调整图像大小至指定尺寸"""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def denoise_image(image, method='gaussian', kernel_size=5, sigma=1.0):
    """
    对图像进行降噪处理
    
    参数:
        image: 输入图像
        method: 降噪方法，可选 'gaussian', 'median', 'bilateral'
        kernel_size: 滤波器核大小
        sigma: 高斯滤波器的标准差
        
    返回:
        降噪后的图像
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif method == 'bilateral':
        # 双边滤波保留边缘的同时去除噪点
        return cv2.bilateralFilter(image, kernel_size, sigma * 15, sigma * 15)
    else:
        return image


def enhance_contrast(image, method='clahe', clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    增强图像对比度
    
    参数:
        image: 输入图像
        method: 对比度增强方法，可选 'clahe', 'histogram_equalization'
        clip_limit: CLAHE限制对比度的阈值
        tile_grid_size: CLAHE的网格大小
        
    返回:
        对比度增强后的图像
    """
    # 如果是彩色图像，在LAB空间中只对L通道进行处理
    if len(image.shape) == 3:
        # 转换到LAB空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
        elif method == 'histogram_equalization':
            l = cv2.equalizeHist(l)
        
        # 合并通道
        lab = cv2.merge((l, a, b))
        # 转回BGR空间
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # 灰度图像直接处理
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image)
        elif method == 'histogram_equalization':
            return cv2.equalizeHist(image)
        
    return image


def sharpen_image(image, kernel_size=3, sigma=1.0, amount=1.5, threshold=0):
    """
    图像锐化处理
    
    参数:
        image: 输入图像
        kernel_size: 高斯核大小
        sigma: 高斯核标准差
        amount: 锐化强度
        threshold: 锐化阈值
        
    返回:
        锐化后的图像
    """
    # 使用高斯模糊创建模糊版本
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # 计算锐化掩码
    sharpened = float(amount + 1) * image - float(amount) * blurred
    
    # 应用阈值
    sharpened = np.maximum(sharpened, np.zeros_like(sharpened))
    sharpened = np.minimum(sharpened, 255 * np.ones_like(sharpened))
    sharpened = sharpened.round().astype(np.uint8)
    
    # 如果有阈值，则只在变化大于阈值的地方应用锐化
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    
    return sharpened


def preprocess_image(image, config):
    """
    根据配置对图像进行预处理
    
    参数:
        image: 输入图像
        config: 预处理配置参数
        
    返回:
        预处理后的图像
    """
    # 复制图像以避免修改原始数据
    processed = image.copy()
    
    # 尺寸调整
    if config['resize']['enabled']:
        processed = resize_image(
            processed, 
            config['resize']['width'], 
            config['resize']['height']
        )
    
    # 去噪
    if config['denoise']['enabled']:
        processed = denoise_image(
            processed,
            method=config['denoise']['method'],
            kernel_size=config['denoise']['kernel_size'],
            sigma=config['denoise']['sigma']
        )
    
    # 对比度增强
    if config['contrast']['enabled']:
        processed = enhance_contrast(
            processed,
            method=config['contrast']['method'],
            clip_limit=config['contrast']['clip_limit'],
            tile_grid_size=config['contrast']['tile_grid_size']
        )
    
    # 锐化
    if config['sharpen']['enabled']:
        processed = sharpen_image(
            processed,
            kernel_size=config['sharpen']['kernel_size'],
            sigma=config['sharpen']['sigma'],
            amount=config['sharpen']['amount'],
            threshold=config['sharpen']['threshold']
        )
    
    return processed