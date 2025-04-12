#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - 通用版本
分割评估指标
"""

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_iou(predicted, ground_truth):
    """
    计算交并比(IoU)
    
    IoU = (重叠区域) / (并集区域)
    
    Args:
        predicted: 预测的二值掩码
        ground_truth: 真实标签的二值掩码
        
    Returns:
        IoU分数（0到1之间的浮点数）
    """
    # 确保二值掩码
    pred_binary = predicted > 0
    gt_binary = ground_truth > 0
    
    # 计算交集和并集
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # 计算IoU，处理除零情况
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_dice_coefficient(predicted, ground_truth):
    """
    计算Dice系数(F1分数)
    
    Dice = 2 * (重叠区域) / (区域之和)
    
    Args:
        predicted: 预测的二值掩码
        ground_truth: 真实标签的二值掩码
        
    Returns:
        Dice系数（0到1之间的浮点数）
    """
    # 确保二值掩码
    pred_binary = predicted > 0
    gt_binary = ground_truth > 0
    
    # 计算交集和面积
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    pred_area = pred_binary.sum()
    gt_area = gt_binary.sum()
    
    # 计算Dice，处理除零情况
    if pred_area + gt_area == 0:
        return 0.0
    
    return 2 * intersection / (pred_area + gt_area)

def calculate_accuracy(predicted, ground_truth):
    """
    计算像素级准确率
    
    准确率 = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        predicted: 预测的二值掩码
        ground_truth: 真实标签的二值掩码
        
    Returns:
        准确率分数（0到1之间的浮点数）
    """
    # 确保二值掩码
    pred_binary = predicted > 0
    gt_binary = ground_truth > 0
    
    # 计算准确率
    correct = (pred_binary == gt_binary).sum()
    total = pred_binary.size
    
    return correct / total

def calculate_precision_recall_f1(predicted, ground_truth):
    """
    计算精确率、召回率和F1分数
    
    精确率 = TP / (TP + FP)
    召回率 = TP / (TP + FN)
    F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
    
    Args:
        predicted: 预测的二值掩码
        ground_truth: 真实标签的二值掩码
        
    Returns:
        (精确率, 召回率, f1)的元组
    """
    # 确保二值掩码
    pred_binary = predicted > 0
    gt_binary = ground_truth > 0
    
    # 展平数组
    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()
    
    # 计算指标，处理除零情况
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def calculate_boundary_f1(predicted, ground_truth, tolerance=2):
    """
    计算边界F1分数
    
    该指标衡量预测掩码边界与真实标签边界的匹配程度。
    
    Args:
        predicted: 预测的二值掩码
        ground_truth: 真实标签的二值掩码
        tolerance: 像素距离容差
        
    Returns:
        边界F1分数（0到1之间的浮点数）
    """
    # 确保二值掩码
    pred_binary = (predicted > 0).astype(np.uint8)
    gt_binary = (ground_truth > 0).astype(np.uint8)
    
    # 查找边界
    pred_boundary = cv2.Canny(pred_binary, 0, 1)
    gt_boundary = cv2.Canny(gt_binary, 0, 1)
    
    # 创建距离图
    pred_dist = cv2.distanceTransform(255 - pred_boundary, cv2.DIST_L2, 3)
    gt_dist = cv2.distanceTransform(255 - gt_boundary, cv2.DIST_L2, 3)
    
    # 计算边界上的像素数
    pred_pixels = np.sum(pred_boundary > 0)
    gt_pixels = np.sum(gt_boundary > 0)
    
    # 如果任一掩码没有边界，返回0
    if pred_pixels == 0 or gt_pixels == 0:
        return 0.0
    
    # 计算容差内的匹配数
    pred_matches = np.sum((gt_dist[pred_boundary > 0] <= tolerance))
    gt_matches = np.sum((pred_dist[gt_boundary > 0] <= tolerance))
    
    # 计算精确率和召回率
    precision = pred_matches / pred_pixels
    recall = gt_matches / gt_pixels
    
    # 计算F1分数
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def calculate_all_metrics(predicted, ground_truth):
    """
    计算所有分割指标
    
    Args:
        predicted: 预测的二值掩码
        ground_truth: 真实标签的二值掩码
        
    Returns:
        指标字典
    """
    iou = calculate_iou(predicted, ground_truth)
    dice = calculate_dice_coefficient(predicted, ground_truth)
    accuracy = calculate_accuracy(predicted, ground_truth)
    precision, recall, f1 = calculate_precision_recall_f1(predicted, ground_truth)
    boundary_f1 = calculate_boundary_f1(predicted, ground_truth)
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'boundary_f1': boundary_f1
    }