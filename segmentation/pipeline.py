#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - Group 13
Flower Segmentation Pipeline Implementation
"""

import cv2
import numpy as np
from .color_spaces import convert_to_hsv, convert_to_lab
from .noise_reduction import apply_gaussian_filter, apply_bilateral_filter
from .thresholding import adaptive_threshold, otsu_threshold
from .morphology import apply_morphology

class FlowerSegmentationPipeline:
    """
    Comprehensive pipeline for flower segmentation that implements
    techniques covered in COMP2032 lectures.
    """
    
    def __init__(self):
        """Initialize the segmentation pipeline with default parameters"""
        # Pipeline parameters - these can be tuned based on dataset characteristics
        self.resize_dim = (256, 256)  # Resize dimensions for consistent processing
        self.use_lab = True  # Whether to use LAB color space (otherwise HSV)
        self.gaussian_kernel = (5, 5)  # Gaussian blur kernel size
        self.bilateral_d = 9  # Bilateral filter diameter
        self.bilateral_sigma_color = 75  # Bilateral filter sigma color
        self.bilateral_sigma_space = 75  # Bilateral filter sigma space
        self.morph_kernel_size = 5  # Morphological operations kernel size
        self.adaptive_block_size = 11  # Block size for adaptive thresholding
        self.adaptive_c = 2  # Constant subtracted from mean in adaptive thresholding
    
    def process(self, image):
        """
        Process an image through the complete segmentation pipeline
        
        Args:
            image: Input RGB image (NumPy array)
            
        Returns:
            segmented_image: Binary image with segmented flower (black background)
            intermediate_results: Dictionary of intermediate processing steps
        """
        # Dictionary to store intermediate results for visualization
        intermediate_results = {}
        
        # Step 1: Resize the image for consistent processing
        resized_image = cv2.resize(image, self.resize_dim, interpolation=cv2.INTER_AREA)
        intermediate_results["1_original_resized"] = resized_image.copy()
        
        # Step 2: Convert to appropriate color space for better segmentation
        if self.use_lab:
            # LAB color space (Lecture 1B - Alternative Color Spaces)
            color_converted = convert_to_lab(resized_image)
            # Extract the A channel which often shows good contrast for flowers
            color_channel = color_converted[:, :, 1].copy()
            intermediate_results["2_lab_a_channel"] = cv2.cvtColor(
                cv2.merge([np.zeros_like(color_channel), color_channel, np.zeros_like(color_channel)]), 
                cv2.COLOR_BGR2GRAY
            )
        else:
            # HSV color space (Lecture 1B - HSV Space)
            color_converted = convert_to_hsv(resized_image)
            # Extract the S (saturation) channel which often separates flower from background
            color_channel = color_converted[:, :, 1].copy()
            intermediate_results["2_hsv_s_channel"] = color_channel
        
        # Step 3: Apply noise reduction (Lecture 3A & 3B - Linear & Non-Linear Filters)
        # First, apply Gaussian filter to reduce noise while preserving edges
        gaussian_filtered = apply_gaussian_filter(color_channel, self.gaussian_kernel)
        intermediate_results["3_gaussian_filtered"] = gaussian_filtered
        
        # Second, apply bilateral filter for edge-preserving smoothing
        bilateral_filtered = apply_bilateral_filter(
            gaussian_filtered, 
            d=self.bilateral_d, 
            sigma_color=self.bilateral_sigma_color, 
            sigma_space=self.bilateral_sigma_space
        )
        intermediate_results["4_bilateral_filtered"] = bilateral_filtered
        
        # Step 4: Apply thresholding to separate foreground from background (Lecture 4 - Thresholding)
        # Try both adaptive and Otsu thresholding
        adaptive_thresh = adaptive_threshold(
            bilateral_filtered, 
            block_size=self.adaptive_block_size, 
            c=self.adaptive_c
        )
        intermediate_results["5_adaptive_threshold"] = adaptive_thresh
        
        otsu_thresh = otsu_threshold(bilateral_filtered)
        intermediate_results["6_otsu_threshold"] = otsu_thresh
        
        # Combine thresholds - use the more conservative approach (where both agree)
        combined_thresh = cv2.bitwise_and(adaptive_thresh, otsu_thresh)
        intermediate_results["7_combined_threshold"] = combined_thresh
        
        # Step 5: Apply morphological operations for cleanup (Lecture 5 - Morphology)
        morph_processed = apply_morphology(
            combined_thresh, 
            kernel_size=self.morph_kernel_size
        )
        intermediate_results["8_morphology"] = morph_processed
        
        # Step 6: Find largest connected component (assuming it's the flower)
        # This is based on Connected Components analysis from Lecture 5
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            morph_processed, connectivity=8
        )
        
        # Skip the background (label 0)
        if num_labels > 1:
            # Find the largest component by area (excluding background)
            max_area = 0
            max_label = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > max_area:
                    max_area = area
                    max_label = i
            
            # Extract the largest component
            largest_component = np.zeros_like(morph_processed)
            largest_component[labels == max_label] = 255
            intermediate_results["9_largest_component"] = largest_component
        else:
            # If no components found, use the morphology result
            largest_component = morph_processed
            intermediate_results["9_largest_component"] = largest_component
        
        # Apply the mask to get the final segmented flower
        segmented_flower = cv2.bitwise_and(
            resized_image, resized_image, mask=largest_component
        )
        intermediate_results["10_segmented_result"] = segmented_flower
        
        # Return the final segmented image and all intermediate results
        return segmented_flower, intermediate_results