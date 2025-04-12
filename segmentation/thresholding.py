#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - Group 13
Thresholding Implementation
Based on Lecture 4 - Thresholding & Binary Images
"""

import cv2
import numpy as np
from skimage import filters

def simple_threshold(image, threshold=127):
    """
    Apply a simple global threshold
    
    As described in Lecture 4, this is the most basic thresholding approach
    where pixels above a fixed threshold are considered foreground.
    
    Args:
        image: Grayscale input image
        threshold: Threshold value (0-255)
        
    Returns:
        Binary image
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def otsu_threshold(image):
    """
    Apply Otsu's automatic thresholding
    
    Otsu's method automatically determines the optimal threshold value
    by minimizing the intra-class variance between foreground and background.
    
    Args:
        image: Grayscale input image
        
    Returns:
        Binary image
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def adaptive_threshold(image, block_size=11, c=2):
    """
    Apply adaptive thresholding
    
    Adaptive thresholding calculates different thresholds for different regions
    of the image, making it more robust to lighting variations.
    
    Args:
        image: Grayscale input image
        block_size: Size of the pixel neighborhood (must be odd)
        c: Constant subtracted from the mean
        
    Returns:
        Binary image
    """
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c
    )

def rosin_threshold(image):
    """
    Apply Rosin's unimodal thresholding
    
    As described in Lecture 4, Rosin's method is effective for unimodal histograms
    where there's no clear bimodal distribution.
    
    Args:
        image: Grayscale input image
        
    Returns:
        Binary image
    """
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    # Find the peak of the histogram
    peak_idx = np.argmax(hist)
    
    # Find the last non-zero bin
    last_idx = 255
    while hist[last_idx] == 0 and last_idx > peak_idx:
        last_idx -= 1
    
    # Calculate the line from peak to last bin
    x1, y1 = peak_idx, hist[peak_idx]
    x2, y2 = last_idx, hist[last_idx]
    
    # Calculate perpendicular distance from each point to the line
    max_dist = 0
    threshold = peak_idx
    
    for i in range(peak_idx + 1, last_idx):
        if hist[i] == 0:
            continue
        
        # Calculate perpendicular distance
        # Line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        dist = abs(a * i + b * hist[i] + c) / np.sqrt(a**2 + b**2)
        
        if dist > max_dist:
            max_dist = dist
            threshold = i
    
    # Apply the computed threshold
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def multi_level_threshold(image, num_classes=3):
    """
    Apply multi-level thresholding for more complex images
    
    This extends beyond simple binary thresholding to separate
    multiple classes (e.g., background, petals, center of flower).
    
    Args:
        image: Grayscale input image
        num_classes: Number of classes to separate
        
    Returns:
        Multi-level thresholded image
    """
    # Use scikit-image's implementation of multi-level Otsu
    thresholds = filters.threshold_multiotsu(image, classes=num_classes)
    
    # Create the result image
    result = np.digitize(image, bins=thresholds)
    
    # Scale result to 0-255 range
    result = (255 * result / (num_classes - 1)).astype(np.uint8)
    
    return result

def local_contrast_adaptative_thresholding(image, window_size=15, k=0.2):
    """
    Apply local contrast adaptive thresholding
    
    This method calculates a threshold based on local contrast:
    T = mean + k * std_dev
    
    Args:
        image: Grayscale input image
        window_size: Size of the local window
        k: Sensitivity parameter
        
    Returns:
        Binary image
    """
    # Calculate local mean
    local_mean = cv2.boxFilter(image, -1, (window_size, window_size), 
                               normalize=True, borderType=cv2.BORDER_REFLECT)
    
    # Calculate local standard deviation
    local_sqr_mean = cv2.boxFilter(image**2, -1, (window_size, window_size), 
                                   normalize=True, borderType=cv2.BORDER_REFLECT)
    local_std = np.sqrt(local_sqr_mean - local_mean**2)
    
    # Calculate local threshold
    local_threshold = local_mean + k * local_std
    
    # Apply threshold
    binary = np.zeros_like(image)
    binary[image > local_threshold] = 255
    
    return binary.astype(np.uint8)