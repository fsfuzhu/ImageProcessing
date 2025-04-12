#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - 通用版本
形态学操作实现
基于讲座5 - 形态学
"""

import cv2
import numpy as np

def apply_erosion(image, kernel_size=3):
    """
    应用形态学腐蚀
    
    腐蚀收缩二值图像的前景（白色）区域。
    它有助于去除小噪点并分离连接的对象。
    
    Args:
        image: 二值输入图像
        kernel_size: 方形结构元素的大小
        
    Returns:
        腐蚀后的二值图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def apply_dilation(image, kernel_size=3):
    """
    应用形态学膨胀
    
    膨胀扩大二值图像的前景（白色）区域。
    它有助于填充小孔洞并连接断开的部分。
    
    Args:
        image: 二值输入图像
        kernel_size: 方形结构元素的大小
        
    Returns:
        膨胀后的二值图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def apply_opening(image, kernel_size=3):
    """
    应用形态学开运算（先腐蚀后膨胀）
    
    开运算去除前景（白色）区域的小对象，
    同时保留较大对象的形状和大小。
    
    Args:
        image: 二值输入图像
        kernel_size: 方形结构元素的大小
        
    Returns:
        开运算后的二值图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def apply_closing(image, kernel_size=3):
    """
    应用形态学闭运算（先膨胀后腐蚀）
    
    闭运算填充前景（白色）区域的小孔洞，
    同时保留对象的形状和大小。
    
    Args:
        image: 二值输入图像
        kernel_size: 方形结构元素的大小
        
    Returns:
        闭运算后的二值图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def apply_advanced_morphology(image, kernel_size=5):
    """
    应用高级形态学操作序列
    
    这个函数应用一系列针对花朵分割优化的形态学操作：
    1. 先应用闭运算填充孔洞
    2. 应用开运算去除小噪声
    3. 应用顶帽变换增强边缘
    
    Args:
        image: 二值输入图像
        kernel_size: 方形结构元素的大小
        
    Returns:
        处理后的二值图像
    """
    # 创建不同大小的核，用于不同操作
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_large = np.ones((kernel_size + 2, kernel_size + 2), np.uint8)
    
    # 首先应用闭运算，填充花朵中小的孔洞
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # 然后应用开运算，去除噪声和小的杂点
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)
    
    # 应用膨胀，确保花朵的连通性
    dilated = cv2.dilate(opened, kernel_small, iterations=1)
    
    # 应用顶帽变换突出小细节
    tophat = cv2.morphologyEx(dilated, cv2.MORPH_TOPHAT, kernel_medium)
    
    # 将顶帽结果添加回去，增强边缘
    enhanced = cv2.add(dilated, tophat)
    
    return enhanced

def find_boundaries(image):
    """
    查找二值图像中对象的边界
    
    这个函数执行形态学梯度（膨胀-腐蚀）来查找对象边界。
    
    Args:
        image: 二值输入图像
        
    Returns:
        带有对象边界的二值图像
    """
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

def fill_holes(image):
    """
    填充二值图像前景区域中的孔洞
    
    这个函数使用漫水填充操作来填充前景中的孔洞。
    
    Args:
        image: 二值输入图像
        
    Returns:
        孔洞已填充的二值图像
    """
    # 复制图像并添加1像素边界
    h, w = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # 创建漫水填充的副本
    filled = image.copy()
    
    # 从边界（背景）填充
    cv2.floodFill(filled, mask, (0, 0), 255)
    
    # 反转填充的图像
    filled_inv = cv2.bitwise_not(filled)
    
    # 与原始图像组合
    result = cv2.bitwise_or(image, filled_inv)
    
    return result