#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - General Version
Thresholding Implementations
Based on Lecture 4 - Thresholding and Binary Images
"""

import cv2
import numpy as np
from skimage import filters

def simple_threshold(image, threshold=127):
    """
    Apply simple global thresholding
    
    As discussed in Lecture 4, this is the most basic form where pixels
    above a fixed threshold are considered foreground.
    
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
    
    Otsu's method automatically determines an optimal threshold value
    by minimizing the intra-class variance between foreground and background.
    
    Args:
        image: Grayscale input image
        
    Returns:
        Binary image
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def adaptive_threshold(image, block_size=15, c=3):
    """
    Apply adaptive thresholding
    
    Adaptive thresholding calculates different thresholds for different regions
    of the image, making it more robust to varying illumination.
    
    Args:
        image: Grayscale input image
        block_size: Size of the pixel neighborhood (must be odd)
        c: Constant subtracted from the mean
        
    Returns:
        Binary image
    """
    # Ensure image is 8-bit grayscale
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Ensure block_size is odd and >= 3
    block_size = max(3, block_size)
    if block_size % 2 == 0:
        block_size += 1
        
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c
    )

def rosin_threshold(image):
    """
    Apply Rosin's unimodal thresholding
    
    As discussed in Lecture 4, Rosin's method is effective for unimodal histograms
    where a clear bimodal distribution is not present.
    
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
    
    # Calculate the line from the peak to the last bin
    x1, y1 = peak_idx, hist[peak_idx]
    x2, y2 = last_idx, hist[last_idx]
    
    # Calculate the perpendicular distance from each point to the line
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
    
    # Apply the calculated threshold
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def multi_level_threshold(image, num_classes=3):
    """
    Apply multi-level thresholding for more complex images
    
    This goes beyond simple binary thresholding and can separate multiple classes
    (e.g., background, flower petals, flower center).
    
    Args:
        image: Grayscale input image
        num_classes: Number of classes to separate
        
    Returns:
        Multi-level thresholded image
    """
    try:
        # Use scikit-image's multi-Otsu implementation
        thresholds = filters.threshold_multiotsu(image, classes=num_classes)
        
        # Create the result image
        result = np.digitize(image, bins=thresholds)
        
        # Scale result to 0-255 range
        result = (255 * result / (num_classes - 1)).astype(np.uint8)
        
        return result
    except Exception as e:
        print(f"Multi-level thresholding failed: {e}")
        # Fallback to Otsu
        return otsu_threshold(image)