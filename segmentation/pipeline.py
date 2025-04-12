#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - 通用版本（简化流程）
花朵分割管线实现 - 以分水岭分割为主要方法
"""

import cv2
import numpy as np
from .color_spaces import convert_to_lab, extract_lab_channels
from .noise_reduction import apply_gaussian_filter, apply_bilateral_filter
from .thresholding import adaptive_threshold, otsu_threshold, multi_level_threshold
from .morphology import fill_holes

class FlowerSegmentationPipeline:
    """
    通用的花朵分割管线，使用分水岭分割作为主要方法
    """
    
    def __init__(self):
        """初始化分割管线，使用通用参数"""
        # 基础参数
        self.resize_dim = (256, 256)  # 调整大小以保持一致的处理
        
        # 增强和滤波参数
        self.gaussian_kernel = (25, 25)  # 高斯滤波核大小
        self.bilateral_d = 9  # 双边滤波直径
        self.bilateral_sigma = 75  # 双边滤波sigma参数
        
        # 阈值参数
        self.adaptive_block_size = 25  # 自适应阈值块大小
        self.adaptive_c = 1  # 自适应阈值常数
        
        # 分水岭参数
        self.min_flower_size_ratio = 0.9  # 最小花朵尺寸比例
    
    def process(self, image):
        """
        处理图像通过简化的分割管线，使用分水岭分割
        
        Args:
            image: 输入的RGB图像（NumPy数组）
            
        Returns:
            segmented_image: 二值图像，带有分割出的花朵（黑色背景）
            intermediate_results: 中间处理步骤的字典
        """
        # 存储中间结果的字典，用于可视化
        intermediate_results = {}
        
        # 步骤1: 调整图像大小以保持一致处理
        resized_image = cv2.resize(image, self.resize_dim, interpolation=cv2.INTER_AREA)
        intermediate_results["1_original_resized"] = resized_image.copy()
        
        # 步骤2: 转换为LAB颜色空间
        lab_image = convert_to_lab(resized_image)
        l, a, b = extract_lab_channels(lab_image)
        
        # 保存LAB通道
        intermediate_results["2a_lab_l_channel"] = l
        intermediate_results["2b_lab_a_channel"] = a
        intermediate_results["2c_lab_b_channel"] = b
        
        # 步骤3: 直方图均衡化增强对比度（基于讲座2）
        enhanced_l = cv2.equalizeHist(l)
        intermediate_results["3a_enhanced_l"] = enhanced_l
        
        # 计算梯度信息（基于讲座6A）
        grad_l = self._calculate_gradient(l)
        grad_a = self._calculate_gradient(a)
        grad_b = self._calculate_gradient(b)
        
        # 合并梯度信息
        combined_gradient = cv2.addWeighted(grad_l, 0.5, cv2.addWeighted(grad_a, 0.5, grad_b, 0.5, 0), 0.5, 0)
        intermediate_results["3b_combined_gradient"] = combined_gradient
        
        # 步骤4: 应用滤波器减少噪声（基于讲座3A和3B）
        # 应用高斯滤波降噪
        gaussian_filtered = apply_gaussian_filter(enhanced_l, self.gaussian_kernel)
        intermediate_results["4a_gaussian_filtered"] = gaussian_filtered
        
        # 应用双边滤波保持边缘
        bilateral_filtered = apply_bilateral_filter(
            gaussian_filtered, 
            d=self.bilateral_d, 
            sigma_color=self.bilateral_sigma, 
            sigma_space=self.bilateral_sigma
        )
        intermediate_results["4b_bilateral_filtered"] = bilateral_filtered
        
        # 步骤5: 多层次阈值处理（基于讲座4）
        # 自适应阈值
        adaptive_thresh = adaptive_threshold(
            bilateral_filtered, 
            block_size=self.adaptive_block_size, 
            c=self.adaptive_c
        )
        intermediate_results["5a_adaptive_threshold"] = adaptive_thresh
        
        # Otsu阈值
        otsu_thresh = otsu_threshold(bilateral_filtered)
        intermediate_results["5b_otsu_threshold"] = otsu_thresh
        
        # 多层次阈值
        multi_thresh = multi_level_threshold(bilateral_filtered, num_classes=3)
        multi_thresh_binary = cv2.threshold(multi_thresh, 127, 255, cv2.THRESH_BINARY)[1]
        intermediate_results["5c_multi_threshold"] = multi_thresh_binary
        
        # 结合不同阈值结果
        combined_thresh = cv2.bitwise_or(
            adaptive_thresh, 
            cv2.bitwise_or(otsu_thresh, multi_thresh_binary)
        )
        intermediate_results["5d_combined_threshold"] = combined_thresh
        
        # 步骤6: 应用分水岭分割（基于讲座7）
        watershed_mask = self._apply_watershed_segmentation(resized_image, combined_thresh)
        intermediate_results["6a_watershed_segmentation"] = watershed_mask
        
        # 填充孔洞（简单的后处理）
        filled_mask = fill_holes(watershed_mask)
        intermediate_results["6b_filled_holes"] = filled_mask
        
        # 找到最大连通区域以移除小的噪声区域
        largest_component = self._extract_largest_component(filled_mask)
        intermediate_results["6c_largest_component"] = largest_component
        
        # 使用分水岭掩码直接获取最终的分割结果
        segmented_flower = cv2.bitwise_and(
            resized_image, resized_image, mask=largest_component
        )
        intermediate_results["7_final_segmented"] = segmented_flower
        
        # 返回最终的分割图像和所有中间结果
        return segmented_flower, intermediate_results
    
    def _calculate_gradient(self, image):
        """计算图像梯度并返回标准化的梯度幅值"""
        # 使用Sobel算子计算梯度
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        magnitude = cv2.magnitude(sobelx, sobely)
        
        # 标准化到0-255范围
        normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized.astype(np.uint8)
    
    def _apply_watershed_segmentation(self, image, binary_mask):
        """应用分水岭算法进行分割"""
        # 确保二值掩码
        _, sure_fg = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 通过距离变换查找确定的前景区域
        dist_transform = cv2.distanceTransform(sure_fg, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # 查找确定的背景区域
        sure_bg = cv2.dilate(binary_mask, np.ones((3,3), np.uint8), iterations=3)
        
        # 计算未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记
        _, markers = cv2.connectedComponents(sure_fg)
        
        # 添加1到所有标签，这样确保背景不是0
        markers = markers + 1
        
        # 将未知区域标记为0
        markers[unknown == 255] = 0
        
        # 应用分水岭算法
        color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 2 else image
        markers = cv2.watershed(color_image, markers.astype(np.int32))
        
        # 创建分水岭结果的掩码
        watershed_mask = np.zeros_like(binary_mask)
        watershed_mask[markers > 1] = 255
        
        return watershed_mask
    
    def _extract_largest_component(self, binary_mask):
        """提取二值掩码中最大的连通区域"""
        # 进行连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # 跳过背景（标签0）
        if num_labels > 1:
            # 按面积找到最大的组件（不包括背景）
            max_area = 0
            max_label = 0
            
            # 图像总像素数
            total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
            min_area = total_pixels * self.min_flower_size_ratio
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > max_area and area > min_area:
                    max_area = area
                    max_label = i
            
            # 提取最大的组件
            largest_component = np.zeros_like(binary_mask)
            if max_label > 0:  # 确保找到了一个符合条件的组件
                largest_component[labels == max_label] = 255
            else:
                # 如果没有找到足够大的组件，使用原始掩码
                largest_component = binary_mask
        else:
            # 如果没有找到组件，使用原始掩码
            largest_component = binary_mask
            
        return largest_component