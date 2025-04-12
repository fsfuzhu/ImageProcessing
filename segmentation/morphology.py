#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - Group 13
Morphological Operations Implementation
Based on Lecture 5 - Morphology
"""

import cv2
import numpy as np

def apply_erosion(image, kernel_size=3):
    """
    Apply morphological erosion
    
    Erosion shrinks the foreground (white) regions of a binary image.
    It's useful for removing small noise and separating connected objects.
    
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
    It's useful for filling small holes and connecting broken parts.
    
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
    
    Opening removes small objects from the foreground (white) regions
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
    
    Closing fills small holes in the foreground (white) regions
    while preserving the shape and size of the objects.
    
    Args:
        image: Binary input image
        kernel_size: Size of the square structuring element
        
    Returns:
        Closed binary image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def apply_morphology(image, kernel_size=5):
    """
    Apply a sequence of morphological operations for optimal cleaning
    
    This function applies a series of morphological operations to
    clean up a binary segmentation mask:
    1. Opening to remove small noise
    2. Closing to fill holes in the main object
    3. Erosion to refine the boundary
    4. Dilation to restore the original size
    
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
    
    # 1. Apply opening to remove small noise
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_small)
    
    # 2. Apply closing to fill holes in the flower
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_large)
    
    # 3. Apply erosion to refine boundaries
    result = cv2.erode(result, kernel_small, iterations=1)
    
    # 4. Apply dilation to restore size
    result = cv2.dilate(result, kernel_small, iterations=1)
    
    return result

def find_boundaries(image):
    """
    Find the boundaries of objects in a binary image
    
    This function performs morphological gradient (dilation - erosion)
    to find object boundaries.
    
    Args:
        image: Binary input image
        
    Returns:
        Binary image with object boundaries
    """
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

def fill_holes(image):
    """
    Fill holes in the foreground regions of a binary image
    
    This function uses floodfill operation to fill holes in the foreground.
    
    Args:
        image: Binary input image
        
    Returns:
        Binary image with holes filled
    """
    # Copy the image and add a 1-pixel border
    h, w = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Create a copy for floodfill
    filled = image.copy()
    
    # Fill from the border (background)
    cv2.floodFill(filled, mask, (0, 0), 255)
    
    # Invert the filled image
    filled_inv = cv2.bitwise_not(filled)
    
    # Combine with the original image
    result = image | filled_inv
    
    return result