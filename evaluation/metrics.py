#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估指标模块，提供IoU、Dice系数等评估指标计算功能
"""

import numpy as np
import cv2


def calculate_iou(mask, ground_truth):
    """
    计算IoU (Intersection over Union)
    
    参数:
        mask: 预测的掩码
        ground_truth: 真实掩码（ground truth）
        
    返回:
        IoU和Dice系数
    """
    # 确保二值图像
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, gt_binary = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
    
    # 转换为布尔掩码
    mask_bool = mask_binary > 0
    gt_bool = gt_binary > 0
    
    # 计算交集和并集
    intersection = np.logical_and(mask_bool, gt_bool).sum()
    union = np.logical_or(mask_bool, gt_bool).sum()
    
    # 计算IoU
    iou = intersection / union if union > 0 else 0.0
    
    # 计算Dice系数
    dice = 2 * intersection / (mask_bool.sum() + gt_bool.sum()) if (mask_bool.sum() + gt_bool.sum()) > 0 else 0.0
    
    return {'iou': iou, 'dice': dice}


def calculate_precision_recall(mask, ground_truth):
    """
    计算精确率和召回率
    
    参数:
        mask: 预测的掩码
        ground_truth: 真实掩码（ground truth）
        
    返回:
        精确率、召回率和F1分数
    """
    # 确保二值图像
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, gt_binary = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
    
    # 转换为布尔掩码
    mask_bool = mask_binary > 0
    gt_bool = gt_binary > 0
    
    # 计算真阳性、假阳性和假阴性
    true_positive = np.logical_and(mask_bool, gt_bool).sum()
    false_positive = np.logical_and(mask_bool, np.logical_not(gt_bool)).sum()
    false_negative = np.logical_and(np.logical_not(mask_bool), gt_bool).sum()
    
    # 计算精确率和召回率
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    
    # 计算F1分数
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}


def calculate_accuracy(mask, ground_truth):
    """
    计算像素级准确率
    
    参数:
        mask: 预测的掩码
        ground_truth: 真实掩码（ground truth）
        
    返回:
        准确率
    """
    # 确保二值图像
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, gt_binary = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
    
    # 转换为布尔掩码
    mask_bool = mask_binary > 0
    gt_bool = gt_binary > 0
    
    # 计算准确率
    correct_pixels = (mask_bool == gt_bool).sum()
    total_pixels = mask_bool.size
    
    accuracy = correct_pixels / total_pixels
    
    return {'accuracy': accuracy}


def calculate_boundary_f1(mask, ground_truth, tolerance=2):
    """
    计算边界F1分数
    
    参数:
        mask: 预测的掩码
        ground_truth: 真实掩码（ground truth）
        tolerance: 边界匹配的容差（像素）
        
    返回:
        边界F1分数
    """
    # 确保二值图像
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, gt_binary = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
    
    # 提取边界
    mask_contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gt_contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not mask_contours or not gt_contours:
        return {'boundary_f1': 0.0}
    
    # 创建边界掩码
    h, w = mask_binary.shape
    mask_boundary = np.zeros((h, w), dtype=np.uint8)
    gt_boundary = np.zeros((h, w), dtype=np.uint8)
    
    cv2.drawContours(mask_boundary, mask_contours, -1, 255, 1)
    cv2.drawContours(gt_boundary, gt_contours, -1, 255, 1)
    
    # 扩张边界以考虑容差
    mask_dilated = cv2.dilate(mask_boundary, np.ones((2*tolerance+1, 2*tolerance+1), np.uint8))
    gt_dilated = cv2.dilate(gt_boundary, np.ones((2*tolerance+1, 2*tolerance+1), np.uint8))
    
    # 计算边界精确率和召回率
    boundary_precision = np.logical_and(mask_boundary > 0, gt_dilated > 0).sum() / (mask_boundary > 0).sum() if (mask_boundary > 0).sum() > 0 else 0.0
    boundary_recall = np.logical_and(gt_boundary > 0, mask_dilated > 0).sum() / (gt_boundary > 0).sum() if (gt_boundary > 0).sum() > 0 else 0.0
    
    # 计算边界F1分数
    boundary_f1 = 2 * (boundary_precision * boundary_recall) / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0.0
    
    return {'boundary_f1': boundary_f1}


def cosine_similarity(feature1, feature2):
    """
    计算余弦相似度
    
    参数:
        feature1: 特征向量1
        feature2: 特征向量2
    
    返回:
        余弦相似度
    """
    # 确保特征是一维数组
    f1 = feature1.flatten()
    f2 = feature2.flatten()
    
    # 计算余弦相似度
    dot_product = np.dot(f1, f2)
    norm_product = np.linalg.norm(f1) * np.linalg.norm(f2)
    
    similarity = dot_product / norm_product if norm_product > 0 else 0.0
    
    return similarity


def calculate_evaluation_metrics(mask, ground_truth):
    """
    计算所有评估指标
    
    参数:
        mask: 预测的掩码
        ground_truth: 真实掩码（ground truth）
        
    返回:
        包含所有指标的字典
    """
    metrics = {}
    
    # 计算IoU和Dice系数
    iou_metrics = calculate_iou(mask, ground_truth)
    metrics.update(iou_metrics)
    
    # 计算精确率、召回率和F1分数
    pr_metrics = calculate_precision_recall(mask, ground_truth)
    metrics.update(pr_metrics)
    
    # 计算准确率
    acc_metrics = calculate_accuracy(mask, ground_truth)
    metrics.update(acc_metrics)
    
    # 计算边界F1分数
    boundary_metrics = calculate_boundary_f1(mask, ground_truth)
    metrics.update(boundary_metrics)
    
    return metrics