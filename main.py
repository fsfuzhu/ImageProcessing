#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - Group 13
Flower Segmentation Main Script
"""

import os
import cv2
import argparse
import glob
import time
from pathlib import Path

from segmentation.pipeline import FlowerSegmentationPipeline
from evaluation.metrics import calculate_iou

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_images(input_dir, output_dir, pipeline_dir, difficulty_levels=None):
    """
    Process all images in the input directory and save results
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save segmentation outputs
        pipeline_dir: Directory to save intermediate pipeline results
        difficulty_levels: List of difficulty subfolder names (e.g., ['easy', 'medium', 'hard'])
    """
    # Initialize the segmentation pipeline
    pipeline = FlowerSegmentationPipeline()
    
    # Default to processing all images if no specific difficulty levels are provided
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
                print(f"Error: Could not read image {img_path}")
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
        print(f"\nProcessed {total_images} images in {total_time:.2f} seconds")
        print(f"Average processing time: {avg_time:.2f} seconds per image")

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
    
    total_iou = 0
    num_images = 0
    
    print("\nEvaluating segmentation results:")
    
    for level in difficulty_levels:
        level_segmented_dir = os.path.join(segmented_dir, level) if level else segmented_dir
        level_gt_dir = os.path.join(ground_truth_dir, level) if level else ground_truth_dir
        
        # Find all segmentation results
        segmented_paths = glob.glob(os.path.join(level_segmented_dir, "*.jpg")) + \
                         glob.glob(os.path.join(level_segmented_dir, "*.png"))
        
        for seg_path in segmented_paths:
            img_filename = os.path.basename(seg_path)
            gt_path = os.path.join(level_gt_dir, img_filename)
            
            # Skip if ground truth doesn't exist
            if not os.path.exists(gt_path):
                print(f"  Warning: Ground truth not found for {img_filename}")
                continue
            
            # Read segmentation result and ground truth
            segmented = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            # Ensure binary masks
            segmented = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)[1]
            ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)[1]
            
            # Calculate IoU
            iou = calculate_iou(segmented, ground_truth)
            print(f"  {img_filename}: IoU = {iou:.4f}")
            
            total_iou += iou
            num_images += 1
    
    # Print overall results
    if num_images > 0:
        avg_iou = total_iou / num_images
        print(f"\nAverage IoU across {num_images} images: {avg_iou:.4f}")

def main():
    """Main function to run the flower segmentation pipeline"""
    parser = argparse.ArgumentParser(description="Flower Segmentation Pipeline")
    parser.add_argument('--input', type=str, default='input-image',
                        help='Directory containing input images')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save segmentation results')
    parser.add_argument('--pipeline', type=str, default='image-processing-pipeline',
                        help='Directory to save intermediate pipeline steps')
    parser.add_argument('--ground-truth', type=str, default='ground-truth',
                        help='Directory containing ground truth masks')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate segmentation results against ground truth')
    
    args = parser.parse_args()
    
    # Define the difficulty levels
    difficulty_levels = ['easy', 'medium', 'hard']
    
    # Process the images
    process_images(args.input, args.output, args.pipeline, difficulty_levels)
    
    # Evaluate results if requested
    if args.evaluate:
        evaluate_segmentation(args.output, args.ground_truth, difficulty_levels)

if __name__ == "__main__":
    main()