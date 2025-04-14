#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - General Version
Flower Segmentation Main Script
"""

import os
import cv2
import argparse
import glob
import time
from pathlib import Path
import numpy as np

from segmentation.pipeline import FlowerSegmentationPipeline
from evaluation.metrics import calculate_iou, calculate_all_metrics

def ensure_dir(directory):
    """Create directory if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_images(input_dir, output_dir, pipeline_dir, difficulty_levels=None):
    """
    Process all images in the input directory and save the results
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save segmentation outputs
        pipeline_dir: Directory to save intermediate pipeline results
        difficulty_levels: List of difficulty subfolder names (e.g., ['easy', 'medium', 'hard'])
    """
    # Initialize the segmentation pipeline
    pipeline = FlowerSegmentationPipeline()
    
    # If no specific difficulty levels are provided, process all images by default
    if difficulty_levels is None:
        difficulty_levels = [""]
    
    total_images = 0
    total_time = 0
    
    # Process each difficulty level
    for level in difficulty_levels:
        level_input_dir = os.path.join(input_dir, level) if level else input_dir
        level_output_dir = os.path.join(output_dir, level) if level else output_dir
        level_pipeline_dir = os.path.join(pipeline_dir, level) if level else pipeline_dir
        
        # Ensure output directories exist
        ensure_dir(level_output_dir)
        ensure_dir(level_pipeline_dir)
        
        # Find all images in the current difficulty level
        image_paths = glob.glob(os.path.join(level_input_dir, "*.jpg")) + \
                      glob.glob(os.path.join(level_input_dir, "*.png"))
        
        print(f"Processing {len(image_paths)} images in {level_input_dir}...")
        
        # Process each image
        for img_path in image_paths:
            img_filename = os.path.basename(img_path)
            img_name = os.path.splitext(img_filename)[0]
            
            print(f"Processing image: {img_filename}")
            
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Failed to read image {img_path}")
                continue
            
            # Time the segmentation process
            start_time = time.time()
            
            # Apply the segmentation pipeline
            segmented_img, intermediate_results = pipeline.process(img)
            
            # Record processing time
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            total_images += 1
            
            print(f"  Processing time: {elapsed_time:.2f} seconds")
            
            # Save the segmentation result
            output_path = os.path.join(level_output_dir, img_filename)
            cv2.imwrite(output_path, segmented_img)
            
            # Save intermediate results
            img_pipeline_dir = os.path.join(level_pipeline_dir, img_name)
            ensure_dir(img_pipeline_dir)
            
            for step_name, result_img in intermediate_results.items():
                step_path = os.path.join(img_pipeline_dir, f"{step_name}.jpg")
                cv2.imwrite(step_path, result_img)
    
    # Print overall statistics
    if total_images > 0:
        avg_time = total_time / total_images
        print(f"Processed {total_images} images in {total_time:.2f} seconds")
        print(f"Average processing time: {avg_time:.2f} seconds/image")

def evaluate_segmentation(segmented_dir, ground_truth_dir, difficulty_levels=None):
    """
    Evaluate segmentation results against ground truth
    
    Args:
        segmented_dir: Directory containing segmentation results
        ground_truth_dir: Directory containing ground truth masks
        difficulty_levels: List of difficulty levels to process
    """
    if difficulty_levels is None:
        difficulty_levels = [""]
    
    total_metrics = {}
    num_images = 0
    
    print("Evaluating segmentation results:")
    
    for level in difficulty_levels:
        level_segmented_dir = os.path.join(segmented_dir, level) if level else segmented_dir
        level_gt_dir = os.path.join(ground_truth_dir, level) if level else ground_truth_dir
        
        # Find all segmentation results
        segmented_paths = glob.glob(os.path.join(level_segmented_dir, "*.jpg")) + \
                         glob.glob(os.path.join(level_segmented_dir, "*.png"))
        
        for seg_path in segmented_paths:
            img_filename = os.path.basename(seg_path)
            # Try finding two possible extensions
            base_name = os.path.splitext(img_filename)[0]
            possible_gt_paths = [
                os.path.join(level_gt_dir, f"{base_name}.png"),
                os.path.join(level_gt_dir, f"{base_name}.jpg")
            ]
            
            # Try to find the existing ground truth file
            gt_path = None
            for path in possible_gt_paths:
                if os.path.exists(path):
                    gt_path = path
                    break
            
            # Skip if ground truth is not found
            if gt_path is None:
                print(f"  Warning: Ground truth not found for {img_filename}")
                continue
            
            # Read segmentation result and ground truth
            segmented = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            # Ensure both masks have the same dimensions
            if segmented.shape != ground_truth.shape:
                print(f"  Resizing ground truth for {img_filename} from {ground_truth.shape} to {segmented.shape}")
                ground_truth = cv2.resize(ground_truth, (segmented.shape[1], segmented.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
            
            # Ensure binary masks
            _, segmented = cv2.threshold(segmented, 1, 255, cv2.THRESH_BINARY)
            _, ground_truth = cv2.threshold(ground_truth, 1, 255, cv2.THRESH_BINARY)
            
            # Calculate all metrics
            metrics = calculate_all_metrics(segmented, ground_truth)
            
            # Initialize the overall metrics dictionary
            if not total_metrics:
                total_metrics = {k: 0 for k in metrics.keys()}
            
            # Accumulate metrics
            for k, v in metrics.items():
                total_metrics[k] += v
            
            print(f"  {img_filename}: IoU = {metrics['iou']:.4f}, Dice = {metrics['dice']:.4f}")
            
            num_images += 1
    
    # Print overall results
    if num_images > 0:
        print(f"Average metrics over {num_images} images:")
        for k, v in total_metrics.items():
            avg_value = v / num_images
            print(f"Average {k}: {avg_value:.4f}")
            
def main():
    """Main function to run the flower segmentation pipeline"""
    parser = argparse.ArgumentParser(description="Flower Segmentation Pipeline")
    parser.add_argument('--input', type=str, default='Dataset_1/input_images',
                        help='Directory containing input images')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save segmentation results')
    parser.add_argument('--pipeline', type=str, default='image-processing-pipeline',
                        help='Directory to save intermediate pipeline steps')
    parser.add_argument('--ground-truth', type=str, default='Dataset_1/ground_truths',
                        help='Directory containing ground truth masks')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate segmentation results against ground truth')
    
    args = parser.parse_args()
    
    # Define difficulty levels
    difficulty_levels = ['easy', 'medium', 'hard']
    
    # Process images
    process_images(args.input, args.output, args.pipeline, difficulty_levels)
    
    # Evaluate results if requested
    if args.evaluate:
        evaluate_segmentation(args.output, args.ground_truth, difficulty_levels)

if __name__ == "__main__":
    main()