#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - Group 13
Image I/O Utilities
"""

import os
import cv2
import numpy as np
import glob
from pathlib import Path

def read_image(image_path, color=True):
    """
    Read an image from disk
    
    Args:
        image_path: Path to the image file
        color: If True, read as BGR; if False, read as grayscale
        
    Returns:
        NumPy array representing the image, or None if reading failed
    """
    if color:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    return img

def save_image(image, output_path):
    """
    Save an image to disk
    
    Args:
        image: NumPy array representing the image
        output_path: Path where the image will be saved
        
    Returns:
        True if successful, False otherwise
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    return cv2.imwrite(output_path, image)

def list_images(directory, extensions=None):
    """
    List all image files in a directory
    
    Args:
        directory: Directory to search for images
        extensions: List of file extensions to include (e.g., ['.jpg', '.png'])
        
    Returns:
        List of paths to image files
    """
    if not os.path.exists(directory):
        return []
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    image_paths = []
    for ext in extensions:
        pattern = os.path.join(directory, f'*{ext}')
        image_paths.extend(glob.glob(pattern))
        
        # Also check for uppercase extensions
        pattern = os.path.join(directory, f'*{ext.upper()}')
        image_paths.extend(glob.glob(pattern))
    
    return sorted(image_paths)

def visualize_results(original, segmented, ground_truth=None, save_path=None):
    """
    Create a visualization comparing original, segmented, and ground truth images
    
    Args:
        original: Original input image
        segmented: Segmentation result
        ground_truth: Ground truth segmentation (optional)
        save_path: Path to save the visualization (optional)
        
    Returns:
        Visualization as NumPy array
    """
    # Ensure all images are in BGR color space
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    if len(segmented.shape) == 2:
        segmented = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    
    if ground_truth is not None and len(ground_truth.shape) == 2:
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2BGR)
    
    # Create the visualization
    if ground_truth is None:
        # Without ground truth, show original and segmented side by side
        height, width = original.shape[:2]
        visualization = np.zeros((height, width * 2, 3), dtype=np.uint8)
        visualization[:, :width] = original
        visualization[:, width:] = segmented
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(visualization, "Segmented", (width + 10, 30), font, 1, (255, 255, 255), 2)
    else:
        # With ground truth, show original, segmented, and ground truth side by side
        height, width = original.shape[:2]
        visualization = np.zeros((height, width * 3, 3), dtype=np.uint8)
        visualization[:, :width] = original
        visualization[:, width:width*2] = segmented
        visualization[:, width*2:] = ground_truth
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(visualization, "Segmented", (width + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(visualization, "Ground Truth", (width*2 + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Save the visualization if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, visualization)
    
    return visualization

def create_folder_structure(base_dir, difficulty_levels=None):
    """
    Create the necessary folder structure for the project
    
    Args:
        base_dir: Base directory
        difficulty_levels: List of difficulty level names
        
    Returns:
        Dictionary of created paths
    """
    paths = {}
    
    # Define the main directories
    dirs = ['input-image', 'output', 'image-processing-pipeline', 'ground-truth']
    
    for dir_name in dirs:
        # Create the main directory
        main_dir = os.path.join(base_dir, dir_name)
        os.makedirs(main_dir, exist_ok=True)
        paths[dir_name] = main_dir
        
        # Create subdirectories for difficulty levels if specified
        if difficulty_levels:
            for level in difficulty_levels:
                level_dir = os.path.join(main_dir, level)
                os.makedirs(level_dir, exist_ok=True)
                paths[f"{dir_name}/{level}"] = level_dir
    
    return paths