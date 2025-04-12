#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - 通用版本
噪声减少实现
基于讲座3A（线性滤波器）和3B（非线性滤波器）
"""

import cv2
import numpy as np

def apply_mean_filter(image, kernel_size=(3, 3)):
    """
    应用均值滤波器进行噪声减少
    
    均值滤波是一种简单的空间滤波器，它用其邻域的均值替换每个像素值。
    这对于去除高斯噪声很有效，但会使边缘模糊。
    
    Args:
        image: 输入图像作为NumPy数组
        kernel_size: 滤波器核的大小，作为元组(宽度, 高度)
        
    Returns:
        滤波后的图像作为NumPy数组
    """
    return cv2.blur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=0):
    """
    应用高斯滤波器进行噪声减少
    
    高斯滤波使用加权平均，其中更靠近中心的像素权重更高。
    这比均值滤波更好地保留边缘。
    
    Args:
        image: 输入图像作为NumPy数组
        kernel_size: 滤波器核的大小，作为元组(宽度, 高度)
        sigma: 高斯的标准差。如果为0，则根据核大小计算。
        
    Returns:
        滤波后的图像作为NumPy数组
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_median_filter(image, kernel_size=3):
    """
    应用中值滤波器进行噪声减少
    
    中值滤波用其邻域的中值替换每个像素。
    它对去除椒盐噪声特别有效，同时保留边缘。
    
    Args:
        image: 输入图像作为NumPy数组
        kernel_size: 方形滤波器核的大小（必须是奇数）
        
    Returns:
        滤波后的图像作为NumPy数组
    """
    return cv2.medianBlur(image, kernel_size)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    应用双边滤波器进行边缘保留平滑
    
    双边滤波同时考虑空间接近度和颜色相似度，
    有效地保留边缘，同时平滑平坦区域。
    
    Args:
        image: 输入图像作为NumPy数组
        d: 每个像素邻域的直径
        sigma_color: 颜色空间中的滤波器sigma
        sigma_space: 坐标空间中的滤波器sigma
        
    Returns:
        滤波后的图像作为NumPy数组
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_anisotropic_diffusion(image, iterations=5, k=25, gamma=0.1):
    """
    应用各向异性扩散滤波器（Perona-Malik）
    
    这是基于讲座3B的各向异性扩散滤波器的实现。
    它在保留边缘的同时平滑图像。
    
    Args:
        image: 输入图像作为NumPy数组
        iterations: 迭代次数
        k: 扩散常数（控制对边缘的敏感度）
        gamma: 控制扩散速度
        
    Returns:
        滤波后的图像作为NumPy数组
    """
    # 转换为float32以进行处理
    img_float = image.astype(np.float32)
    
    # 创建输出图像
    out = img_float.copy()
    
    # 定义扩散的4邻域结构
    delta = [
        (0, 1),   # 右
        (0, -1),  # 左
        (1, 0),   # 下
        (-1, 0)   # 上
    ]
    
    # 应用扩散迭代
    for _ in range(iterations):
        # 计算每个方向的扩散
        diffusion = np.zeros_like(out)
        
        for dx, dy in delta:
            # 计算移位数组
            shifted = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            
            # 计算当前图像和移位图像之间的差异
            diff = shifted - out
            
            # 应用扩散函数（Perona-Malik）
            # g(x) = 1 / (1 + (x/k)^2)
            g = 1.0 / (1.0 + (diff / k) ** 2)
            
            # 累积扩散
            diffusion += g * diff
        
        # 更新图像
        out += gamma * diffusion
    
    # 转换回uint8
    return np.clip(out, 0, 255).astype(np.uint8)