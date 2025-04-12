#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - Group 13
Segmentation Evaluation Metrics
"""

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_iou(predicted, ground_truth):
    """
    Calculate Intersection over Union (IoU)
    
    IoU = (Area of Overlap) / (Area of Union)
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        
    Returns:
        IoU score (float between 0 and 1)
    """
    # Ensure binary masks
    pred_binary = predicted > 0
    gt_binary = ground_truth > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # Calculate IoU, handle division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_dice_coefficient(predicted, ground_truth):
    """
    Calculate Dice coefficient (F1 score)
    
    Dice = 2 * (Area of Overlap) / (Sum of Areas)
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        
    Returns:
        Dice coefficient (float between 0 and 1)
    """
    # Ensure binary masks
    pred_binary = predicted > 0
    gt_binary = ground_truth > 0
    
    # Calculate intersection and areas
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    pred_area = pred_binary.sum()
    gt_area = gt_binary.sum()
    
    # Calculate Dice, handle division by zero
    if pred_area + gt_area == 0:
        return 0.0
    
    return 2 * intersection / (pred_area + gt_area)

def calculate_accuracy(predicted, ground_truth):
    """
    Calculate pixel-wise accuracy
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        
    Returns:
        Accuracy score (float between 0 and 1)
    """
    # Ensure binary masks
    pred_binary = predicted > 0
    gt_binary = ground_truth > 0
    
    # Calculate accuracy
    correct = (pred_binary == gt_binary).sum()
    total = pred_binary.size
    
    return correct / total

def calculate_precision_recall_f1(predicted, ground_truth):
    """
    Calculate precision, recall, and F1 score
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    # Ensure binary masks
    pred_binary = predicted > 0
    gt_binary = ground_truth > 0
    
    # Flatten the arrays
    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()
    
    # Calculate metrics, handle division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def calculate_boundary_f1(predicted, ground_truth, tolerance=2):
    """
    Calculate boundary F1 score
    
    This metric measures how well the predicted mask boundaries
    match with the ground truth boundaries.
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        tolerance: Distance tolerance in pixels
        
    Returns:
        Boundary F1 score (float between 0 and 1)
    """
    # Ensure binary masks
    pred_binary = (predicted > 0).astype(np.uint8)
    gt_binary = (ground_truth > 0).astype(np.uint8)
    
    # Find boundaries
    pred_boundary = cv2.Canny(pred_binary, 0, 1)
    gt_boundary = cv2.Canny(gt_binary, 0, 1)
    
    # Create distance maps
    pred_dist = cv2.distanceTransform(255 - pred_boundary, cv2.DIST_L2, 3)
    gt_dist = cv2.distanceTransform(255 - gt_boundary, cv2.DIST_L2, 3)
    
    # Count pixels on boundaries
    pred_pixels = np.sum(pred_boundary > 0)
    gt_pixels = np.sum(gt_boundary > 0)
    
    # If either mask has no boundary, return 0
    if pred_pixels == 0 or gt_pixels == 0:
        return 0.0
    
    # Count matches within tolerance
    pred_matches = np.sum((gt_dist[pred_boundary > 0] <= tolerance))
    gt_matches = np.sum((pred_dist[gt_boundary > 0] <= tolerance))
    
    # Calculate precision and recall
    precision = pred_matches / pred_pixels
    recall = gt_matches / gt_pixels
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def calculate_all_metrics(predicted, ground_truth):
    """
    Calculate all segmentation metrics
    
    Args:
        predicted: Predicted binary mask
        ground_truth: Ground truth binary mask
        
    Returns:
        Dictionary of metrics
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