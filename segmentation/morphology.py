#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - General Version
Morphological Operation Implementations
Based on Lecture 5 - Morphology
"""

import cv2
import numpy as np

def apply_erosion(image, kernel_size=3):
    """
    Apply morphological erosion
    
    Erosion shrinks the foreground (white) regions of a binary image.
    It helps remove small noise and detach connected objects.
    
    Args:
        image: Binary input image
        kernel_size: Size of the square structuring element
        
    Returns:
        Eroded binary image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def apply_dilation(image, kernel_size=3):
    """
    Apply morphological dilation
    
    Dilation expands the foreground (white) regions of a binary image.
    It helps fill small holes and connect broken parts.
    
    Args:
        image: Binary input image
        kernel_size: Size of the square structuring element
        
    Returns:
        Dilated binary image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def apply_opening(image, kernel_size=3):
    """
    Apply morphological opening (erosion followed by dilation)
    
    Opening removes small objects from the foreground (white) areas
    while preserving the shape and size of larger objects.
    
    Args:
        image: Binary input image
        kernel_size: Size of the square structuring element
        
    Returns:
        Opened binary image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def apply_closing(image, kernel_size=3):
    """
    Apply morphological closing (dilation followed by erosion)
    
    Closing fills small holes in the foreground (white) areas
    while preserving the shape and size of objects.
    
    Args:
        image: Binary input image
        kernel_size: Size of the square structuring element
        
    Returns:
        Closed binary image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def apply_advanced_morphology(image, kernel_size=5):
    """
    Apply an advanced sequence of morphological operations
    
    This function applies a series of operations tailored for flower segmentation:
    1. Closing to fill holes first
    2. Opening to remove small noise
    3. Top-hat transform to enhance edges
    
    Args:
        image: Binary input image
        kernel_size: Size of the square structuring element
        
    Returns:
        Processed binary image
    """
    # Create kernels of different sizes for different operations
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_large = np.ones((kernel_size + 2, kernel_size + 2), np.uint8)
    
    # Apply closing first to fill small holes within the flower
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # Then apply opening to remove noise and small speckles
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)
    
    # Apply dilation to ensure connectivity of the flower parts
    dilated = cv2.dilate(opened, kernel_small, iterations=1)
    
    # Apply Top-hat transform to highlight small details
    tophat = cv2.morphologyEx(dilated, cv2.MORPH_TOPHAT, kernel_medium)
    
    # Add the tophat result back to enhance edges
    enhanced = cv2.add(dilated, tophat)
    
    return enhanced

def find_boundaries(image):
    """
    Find the boundaries of objects in a binary image
    
    This performs morphological gradient (dilation - erosion) to find boundaries.
    
    Args:
        image: Binary input image
        
    Returns:
        Binary image with object boundaries
    """
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

def fill_holes(image):
    """
    Fill holes within the foreground regions of a binary image
    
    This uses flood fill to fill holes in the foreground.
    
    Args:
        image: Binary input image
        
    Returns:
        Binary image with holes filled
    """
    # Copy the image and add a 1-pixel border
    h, w = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Create a copy for flood fill
    filled = image.copy()
    
    # Flood fill from the border (background)
    cv2.floodFill(filled, mask, (0, 0), 255)
    
    # Invert the filled image
    filled_inv = cv2.bitwise_not(filled)
    
    # Combine with the original image
    result = cv2.bitwise_or(image, filled_inv)
    
    return result