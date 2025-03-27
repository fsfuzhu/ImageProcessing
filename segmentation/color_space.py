#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
颜色空间转换模块，提供RGB, HSV, LAB, YCrCb等颜色空间转换功能
"""

import cv2
import numpy as np


def convert_color_space(image, target_space='hsv'):
    """
    将图像转换到指定的颜色空间
    
    参数:
        image: 输入图像 (BGR格式)
        target_space: 目标颜色空间，可选 'rgb', 'hsv', 'lab', 'ycrcb'
        
    返回:
        转换后的图像和各个通道
    """
    # 确保图像是3通道
    if len(image.shape) != 3:
        # 如果是灰度图，转换为3通道
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 根据目标颜色空间进行转换
    if target_space.lower() == 'rgb':
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 分割通道 - R, G, B
        channels = cv2.split(converted)
        return converted
    
    elif target_space.lower() == 'hsv':
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 分割通道 - H, S, V
        channels = cv2.split(converted)
        return converted
    
    elif target_space.lower() == 'lab':
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # 分割通道 - L, a, b
        channels = cv2.split(converted)
        return converted
    
    elif target_space.lower() == 'ycrcb':
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # 分割通道 - Y, Cr, Cb
        channels = cv2.split(converted)
        return converted
    
    else:
        # 默认返回原始图像
        return image


def extract_color_features(image, target_space='hsv'):
    """
    从指定颜色空间提取颜色特征
    
    参数:
        image: 输入图像
        target_space: 目标颜色空间
        
    返回:
        各个通道的特征统计信息
    """
    # 转换颜色空间
    converted = convert_color_space(image, target_space)
    
    # 分割通道
    channels = cv2.split(converted)
    
    # 提取每个通道的特征
    features = []
    for i, channel in enumerate(channels):
        # 计算直方图
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # 计算基本统计量
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        median_val = np.median(channel)
        
        # 保存特征
        channel_features = {
            'histogram': hist,
            'mean': mean_val,
            'std': std_val,
            'median': median_val
        }
        features.append(channel_features)
    
    return features


def adaptive_color_threshold(image, target_space='hsv', method='otsu'):
    """
    对颜色通道进行自适应阈值处理
    
    参数:
        image: 输入图像
        target_space: 目标颜色空间
        method: 阈值方法，可选 'otsu', 'triangle', 'adaptive'
        
    返回:
        处理后的二值图像
    """
    # 转换颜色空间
    converted = convert_color_space(image, target_space)
    
    # 分割通道
    channels = cv2.split(converted)
    
    # 初始化掩码
    mask = np.zeros_like(channels[0])
    
    # 根据目标颜色空间选择适当的通道进行处理
    if target_space.lower() == 'hsv':
        # HSV: 使用饱和度和亮度通道
        sat_channel = channels[1]
        val_channel = channels[2]
        
        # 对饱和度通道进行阈值处理
        if method == 'otsu':
            _, sat_mask = cv2.threshold(sat_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'triangle':
            _, sat_mask = cv2.threshold(sat_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        elif method == 'adaptive':
            sat_mask = cv2.adaptiveThreshold(sat_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        
        # 对亮度通道进行阈值处理
        if method == 'otsu':
            _, val_mask = cv2.threshold(val_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'triangle':
            _, val_mask = cv2.threshold(val_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        elif method == 'adaptive':
            val_mask = cv2.adaptiveThreshold(val_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        
        # 结合两个掩码
        mask = cv2.bitwise_and(sat_mask, val_mask)
    
    elif target_space.lower() == 'lab':
        # LAB: 使用a和b通道
        a_channel = channels[1]
        b_channel = channels[2]
        
        # 对a通道进行阈值处理
        if method == 'otsu':
            _, a_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'triangle':
            _, a_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        elif method == 'adaptive':
            a_mask = cv2.adaptiveThreshold(a_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        
        # 对b通道进行阈值处理
        if method == 'otsu':
            _, b_mask = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'triangle':
            _, b_mask = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        elif method == 'adaptive':
            b_mask = cv2.adaptiveThreshold(b_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        
        # 结合两个掩码
        mask = cv2.bitwise_and(a_mask, b_mask)
    
    else:
        # 对于其他颜色空间，默认使用所有通道的综合
        channel_masks = []
        for channel in channels:
            if method == 'otsu':
                _, channel_mask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == 'triangle':
                _, channel_mask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            elif method == 'adaptive':
                channel_mask = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)
            channel_masks.append(channel_mask)
        
        # 结合所有通道掩码
        mask = channel_masks[0]
        for channel_mask in channel_masks[1:]:
            mask = cv2.bitwise_and(mask, channel_mask)
    
    return mask