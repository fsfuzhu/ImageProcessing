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
from evaluation.evaluate import evaluate_segmentation_with_red_mask

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
    parser.add_argument('--evaluate-output', type=str, default='evaluation_output',
                        help='Directory to save evaluation results')
    parser.add_argument('--using-difficulty', action='store_true',
                        help='Use/skip difficulty levels for dataset1/dataset2')
    
    args = parser.parse_args()
    
    # Define difficulty levels
    if args.using_difficulty:
        difficulty_levels = ['easy', 'medium', 'hard']
    else:
        # If not using difficulty levels, process all images in the root directory
        difficulty_levels = None
    
    # Process images
    process_images(args.input, args.output, args.pipeline, difficulty_levels)
    
    # Evaluate results if requested
    if args.evaluate:
        print("Evaluating segmentation using red mask extraction...")
        evaluate_segmentation_with_red_mask(
            args.input, args.output, args.ground_truth, 
            difficulty_levels, args.evaluate_output
        )

if __name__ == "__main__":
    main()