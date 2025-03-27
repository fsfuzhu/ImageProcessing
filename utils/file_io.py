#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件输入输出工具模块，提供文件读写、目录管理等功能
"""

import os
import glob
import cv2
import numpy as np
import json


def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_filename(file_path, with_extension=False):
    """
    获取文件名（不包含目录和扩展名）
    
    参数:
        file_path: 文件路径
        with_extension: 是否包含扩展名
        
    返回:
        文件名
    """
    if with_extension:
        return os.path.basename(file_path)
    else:
        return os.path.splitext(os.path.basename(file_path))[0]


def list_files(directory, extensions=None):
    """
    列出目录中指定扩展名的文件
    
    参数:
        directory: 目录路径
        extensions: 扩展名列表，例如 ['.jpg', '.png']
        
    返回:
        文件路径列表
    """
    if not os.path.exists(directory):
        return []
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        file_list.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    
    return sorted(file_list)


def read_image(file_path, flags=cv2.IMREAD_COLOR):
    """
    读取图像文件
    
    参数:
        file_path: 图像文件路径
        flags: OpenCV imread标志
        
    返回:
        图像数据，如果读取失败则返回None
    """
    if not os.path.exists(file_path):
        print(f"Error: Image file not found: {file_path}")
        return None
    
    image = cv2.imread(file_path, flags)
    if image is None:
        print(f"Error: Failed to read image: {file_path}")
    
    return image


def save_image(file_path, image, create_dirs=True):
    """
    保存图像文件
    
    参数:
        file_path: 图像文件保存路径
        image: 图像数据
        create_dirs: 是否创建目录（如果不存在）
        
    返回:
        是否保存成功
    """
    if image is None:
        print(f"Error: Cannot save None image to: {file_path}")
        return False
    
    if create_dirs:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
    
    return cv2.imwrite(file_path, image)


def save_metrics(file_path, metrics):
    """
    保存评估指标为JSON文件
    
    参数:
        file_path: JSON文件保存路径
        metrics: 评估指标字典
        
    返回:
        是否保存成功
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return True
    except Exception as e:
        print(f"Error saving metrics to {file_path}: {str(e)}")
        return False


def load_metrics(file_path):
    """
    从JSON文件加载评估指标
    
    参数:
        file_path: JSON文件路径
        
    返回:
        评估指标字典，如果加载失败则返回None
    """
    if not os.path.exists(file_path):
        print(f"Error: Metrics file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            metrics = json.load(f)
        
        return metrics
    except Exception as e:
        print(f"Error loading metrics from {file_path}: {str(e)}")
        return None


def get_matching_files(source_dir, target_dir, extensions=None):
    """
    获取两个目录中具有相同文件名的文件
    
    参数:
        source_dir: 源目录
        target_dir: 目标目录
        extensions: 扩展名列表
        
    返回:
        匹配文件的列表，每项为(源文件路径, 目标文件路径)
    """
    if not os.path.exists(source_dir) or not os.path.exists(target_dir):
        return []
    
    # 获取源目录中的文件
    source_files = list_files(source_dir, extensions)
    source_filenames = {get_filename(f): f for f in source_files}
    
    # 获取目标目录中的文件
    target_files = list_files(target_dir, extensions)
    target_filenames = {get_filename(f): f for f in target_files}
    
    # 查找匹配的文件
    matching_files = []
    for name, source_path in source_filenames.items():
        if name in target_filenames:
            matching_files.append((source_path, target_filenames[name]))
    
    return matching_files


def create_directory_structure(base_dir, subdirs=None):
    """
    创建目录结构
    
    参数:
        base_dir: 基础目录
        subdirs: 子目录列表
    """
    ensure_dir(base_dir)
    
    if subdirs:
        for subdir in subdirs:
            ensure_dir(os.path.join(base_dir, subdir))