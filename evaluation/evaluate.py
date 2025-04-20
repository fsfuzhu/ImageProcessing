#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - Flower Segmentation Evaluation Module
Evaluates segmentation results using red masks from ground truth images
"""

import os
import cv2
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

def extract_red_mask(image_path):
    """
    Extract red mask from ground truth image
    
    Args:
        image_path: Path to ground truth image
        
    Returns:
        Binary mask where red regions are 255, background is 0
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    
    # Convert to HSV color space for easier color separation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for red color
    # Red has two ranges in HSV (0-10 and 170-180)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine masks
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    return red_mask

def apply_mask_to_original(original_image_path, mask):
    """
    Apply mask to original image, keeping the flower and making background black
    
    Args:
        original_image_path: Path to original image
        mask: Binary mask
        
    Returns:
        Processed image with flower foreground and black background
    """
    # Read original image
    original = cv2.imread(original_image_path)
    if original is None:
        print(f"Error: Cannot read original image {original_image_path}")
        return None
    
    # Ensure mask is binary
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # Resize mask if original and mask have different dimensions
    if original.shape[:2] != binary_mask.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (original.shape[1], original.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
    
    # Create 3-channel mask
    mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    # Apply mask to original image
    result = cv2.bitwise_and(original, mask_3ch)
    
    return result

def calculate_iou(predicted_mask, ground_truth_mask):
    """
    Calculate IoU (Intersection over Union)
    
    Args:
        predicted_mask: Predicted binary mask
        ground_truth_mask: Ground truth binary mask
        
    Returns:
        IoU value (float between 0-1)
    """
    # Ensure masks are binary
    _, pred_binary = cv2.threshold(predicted_mask, 1, 1, cv2.THRESH_BINARY)
    _, gt_binary = cv2.threshold(ground_truth_mask, 1, 1, cv2.THRESH_BINARY)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_dice_coefficient(predicted_mask, ground_truth_mask):
    """
    Calculate Dice coefficient (F1 score)
    
    Args:
        predicted_mask: Predicted binary mask
        ground_truth_mask: Ground truth binary mask
        
    Returns:
        Dice coefficient (float between 0-1)
    """
    # Ensure masks are binary
    _, pred_binary = cv2.threshold(predicted_mask, 1, 1, cv2.THRESH_BINARY)
    _, gt_binary = cv2.threshold(ground_truth_mask, 1, 1, cv2.THRESH_BINARY)
    
    # Calculate intersection
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    
    # Calculate sizes of both masks
    size_pred = pred_binary.sum()
    size_gt = gt_binary.sum()
    
    # Avoid division by zero
    if size_pred + size_gt == 0:
        return 0.0
    
    # Calculate Dice coefficient
    return 2 * intersection / (size_pred + size_gt)
def compute_cosine_similarity(segmented_image, ground_truth):
    """
    Calculate cosine similarity between two images
    
    Args:
        image1: First image as NumPy array
        image2: Second image as NumPy array
        
    Returns:
        Cosine similarity value (float between 0-1)
    """
    # Ensure images have the same shape
    if segmented_image.shape != ground_truth.shape:
        segmented_image = cv2.resize(segmented_image, 
                                    (ground_truth.shape[1], ground_truth.shape[0]))
    
    # Flatten images to 1D vectors
    seg_vector = segmented_image.flatten().astype(np.float32)
    gt_vector = ground_truth.flatten().astype(np.float32)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([seg_vector], [gt_vector])[0][0]
    
    return similarity
def evaluate_segmentation_with_red_mask(input_dir, segmented_dir, ground_truth_dir, difficulty_levels=None, output_dir=None):
    """
    Evaluate segmentation results using red masks from ground truth images
    
    Args:
        input_dir: Directory with original input images
        segmented_dir: Directory with segmentation results
        ground_truth_dir: Directory with ground truth images
        difficulty_levels: List of difficulty levels (e.g. ['easy', 'medium', 'hard'])
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # If no difficulty levels specified, process all images as default
    if difficulty_levels is None:
        difficulty_levels = [""]
    
    # Ensure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results list and cumulative metrics
    all_results = []
    total_iou = 0
    total_accuracy = 0
    total_dice = 0
    total_cosine_sim = 0
    total_images = 0
    
    print("Evaluating segmentation results:")
    
    # Process each difficulty level
    for level in difficulty_levels:
        # Build directory paths for the current difficulty level
        level_input_dir = os.path.join(input_dir, level) if level else input_dir
        level_segmented_dir = os.path.join(segmented_dir, level) if level else segmented_dir
        level_gt_dir = os.path.join(ground_truth_dir, level) if level else ground_truth_dir
        
        if output_dir:
            level_output_dir = os.path.join(output_dir, level) if level else output_dir
            os.makedirs(level_output_dir, exist_ok=True)
        
        # Find all segmentation results
        segmented_paths = glob.glob(os.path.join(level_segmented_dir, "*.jpg")) + \
                          glob.glob(os.path.join(level_segmented_dir, "*.png"))
        
        print(f"Processing {len(segmented_paths)} images, difficulty level: {level or 'default'}")
        
        # Process each image
        for seg_path in segmented_paths:
            # Get image filename
            img_filename = os.path.basename(seg_path)
            base_name = os.path.splitext(img_filename)[0]
            
            # Find corresponding ground truth file
            possible_gt_paths = [
                os.path.join(level_gt_dir, f"{base_name}.png"),
                os.path.join(level_gt_dir, f"{base_name}.jpg")
            ]
            
            gt_path = None
            for path in possible_gt_paths:
                if os.path.exists(path):
                    gt_path = path
                    break
            
            # Skip if ground truth not found
            if gt_path is None:
                print(f"  Warning: Ground truth not found for {img_filename}")
                continue
                
            # Find corresponding original input image
            possible_input_paths = [
                os.path.join(level_input_dir, f"{base_name}.png"),
                os.path.join(level_input_dir, f"{base_name}.jpg")
            ]
            
            input_path = None
            for path in possible_input_paths:
                if os.path.exists(path):
                    input_path = path
                    break
            
            # Skip if original image not found
            if input_path is None:
                print(f"  Warning: Original input image not found for {img_filename}")
                continue
            
            print(f"  Processing {img_filename}...")
            
            try:
                # 1. Extract red mask from ground truth
                red_mask = extract_red_mask(gt_path)
                if red_mask is None:
                    continue
                
                # 2. Read segmentation result
                segmented = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                if segmented is None:
                    print(f"  Error: Cannot read segmentation result {seg_path}")
                    continue
                
                # 3. Apply red mask to original image to get ground truth flower
                ground_truth_flower = apply_mask_to_original(input_path, red_mask)
                if ground_truth_flower is None:
                    continue
                
                # 4. Ensure segmentation result and ground truth have same dimensions
                if segmented.shape != red_mask.shape:
                    segmented = cv2.resize(segmented, (red_mask.shape[1], red_mask.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                
                # 5. Binarize segmentation result
                _, binary_segmented = cv2.threshold(segmented, 1, 255, cv2.THRESH_BINARY)
                
                # 6. Calculate evaluation metrics
                iou = calculate_iou(binary_segmented, red_mask)
                
                # Calculate accuracy
                binary_segmented_flat = (binary_segmented > 0).flatten()
                red_mask_flat = (red_mask > 0).flatten()
                accuracy = accuracy_score(red_mask_flat, binary_segmented_flat)
                
                # Calculate Dice coefficient
                dice = calculate_dice_coefficient(binary_segmented, red_mask)
                # Calculate cosine similarity
                cosine_sim = compute_cosine_similarity(binary_segmented, red_mask)
                # 7. Add to results list
                result = {
                    'image': img_filename,
                    'difficulty': level or 'default',
                    'iou': iou,
                    'accuracy': accuracy,
                    'dice': dice,
                    'cosine_similarity': cosine_sim  # Add this line
                }
                all_results.append(result)
                
                # 8. Accumulate metrics
                total_iou += iou
                total_accuracy += accuracy
                total_dice += dice
                total_cosine_sim += cosine_sim
                total_images += 1
                
                print(f"  {img_filename}: IoU = {iou:.4f}, Accuracy = {accuracy:.4f}, Dice = {dice:.4f}, Cosine Similarity = {cosine_sim:.4f}")  
                # 9. Save visualization if output directory specified
                if output_dir:
                    # Create visualization comparison
                    vis_path = os.path.join(level_output_dir, f"{base_name}_comparison.png")
                    
                    # Original image
                    original = cv2.imread(input_path)
                    
                    # Segmentation result
                    segmented_color = cv2.imread(seg_path)
                    
                    # Create visualization image
                    # Ensure all images have the same size
                    h, w = original.shape[:2]
                    segmented_resized = cv2.resize(segmented_color, (w, h))
                    ground_truth_resized = cv2.resize(ground_truth_flower, (w, h))
                    
                    # Horizontally stack original, segmentation result, and ground truth
                    comparison = np.hstack((original, segmented_resized, ground_truth_resized))
                    
                    # Add text labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    y_pos = 30
                    cv2.putText(comparison, "Original", (10, y_pos), font, 1, (255, 255, 255), 2)
                    cv2.putText(comparison, "Segmented", (w + 10, y_pos), font, 1, (255, 255, 255), 2)
                    cv2.putText(comparison, "Ground Truth", (2*w + 10, y_pos), font, 1, (255, 255, 255), 2)
                    cv2.putText(comparison, f"IoU: {iou:.4f}", (10, h - 10), font, 0.7, (255, 255, 255), 2)
                    
                    # Save comparison image
                    cv2.imwrite(vis_path, comparison)
                
            except Exception as e:
                print(f"  Error processing {img_filename}: {e}")
    
    # Calculate average metrics
    avg_metrics = {}
    # In the section where avg_metrics is calculated:
    if total_images > 0:
        avg_metrics['avg_iou'] = total_iou / total_images
        avg_metrics['avg_accuracy'] = total_accuracy / total_images
        avg_metrics['avg_dice'] = total_dice / total_images
        avg_metrics['avg_cosine_similarity'] = total_cosine_sim / total_images  # Add this line
        avg_metrics['total_images'] = total_images
        
        print(f"\nEvaluation complete, processed {total_images} images")
        print(f"Average IoU: {avg_metrics['avg_iou']:.4f}")
        print(f"Average Accuracy: {avg_metrics['avg_accuracy']:.4f}")
        print(f"Average Dice Coefficient: {avg_metrics['avg_dice']:.4f}")
        print(f"Average Cosine Similarity: {avg_metrics['avg_cosine_similarity']:.4f}")  
    
    # Save evaluation results if output directory specified
    if output_dir and all_results:
        # Save as CSV
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, "segmentation_evaluation.csv")
        results_df.to_csv(csv_path, index=False)
        
        # Save summary report
        summary_path = os.path.join(output_dir, "evaluation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Flower Segmentation Evaluation Results\n")
            f.write("==================================\n\n")
            f.write(f"Number of images evaluated: {total_images}\n\n")
            f.write("Average metrics:\n")
            f.write(f"- IoU: {avg_metrics['avg_iou']:.4f}\n")
            f.write(f"- Accuracy: {avg_metrics['avg_accuracy']:.4f}\n")
            f.write(f"- Dice Coefficient: {avg_metrics['avg_dice']:.4f}\n")
            f.write(f"- Cosine Similarity: {avg_metrics['avg_cosine_similarity']:.4f}\n\n")
            
            # Group by difficulty level
            if len(difficulty_levels) > 1:
                f.write("Metrics by difficulty level:\n")
                for level in difficulty_levels:
                    level_results = [r for r in all_results if r['difficulty'] == (level or 'default')]
                    if level_results:
                        level_iou = sum(r['iou'] for r in level_results) / len(level_results)
                        level_acc = sum(r['accuracy'] for r in level_results) / len(level_results)
                        level_dice = sum(r['dice'] for r in level_results) / len(level_results)
                        level_cosine_sim = sum(r['cosine_similarity'] for r in level_results) / len(level_results)
                        
                        f.write(f"\n{level or 'default'} level ({len(level_results)} images):\n")
                        f.write(f"- IoU: {level_iou:.4f}\n")
                        f.write(f"- Accuracy: {level_acc:.4f}\n")
                        f.write(f"- Dice Coefficient: {level_dice:.4f}\n")
                        f.write(f"- Cosine Similarity: {level_cosine_sim:.4f}\n")
        
        print(f"Evaluation results saved to {output_dir}")
    
    return avg_metrics