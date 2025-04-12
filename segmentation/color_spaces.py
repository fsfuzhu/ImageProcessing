#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - Group 13
Color Space Conversion Utilities
Based on Lecture 1B - Digital Images and Point Processes
"""

import cv2
import numpy as np

def convert_to_hsv(image):
    """
    Convert an RGB image to HSV color space
    
    HSV separates color (hue) from intensity (value) making it less sensitive
    to illumination changes, which is beneficial for segmentation.
    
    Args:
        image: RGB image as NumPy array
        
    Returns:
        HSV image as NumPy array
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def convert_to_lab(image):
    """
    Convert an RGB image to LAB color space
    
    LAB separates lightness (L) from color information (A and B channels),
    making it useful for distinguishing color differences regardless of brightness.
    
    Args:
        image: RGB image as NumPy array
        
    Returns:
        LAB image as NumPy array
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

def convert_to_grayscale(image):
    """
    Convert an RGB image to grayscale using the weighted formula
    
    As mentioned in Lecture 1B, we can convert RGB to grayscale using:
    I = 0.30R + 0.59G + 0.11B (weights based on human eye sensitivity)
    
    Args:
        image: RGB image as NumPy array
        
    Returns:
        Grayscale image as NumPy array
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def extract_hsv_channels(hsv_image):
    """
    Extract individual channels from an HSV image
    
    Args:
        hsv_image: HSV image as NumPy array
        
    Returns:
        Tuple of (H, S, V) channels as separate NumPy arrays
    """
    h, s, v = cv2.split(hsv_image)
    return h, s, v

def extract_lab_channels(lab_image):
    """
    Extract individual channels from a LAB image
    
    Args:
        lab_image: LAB image as NumPy array
        
    Returns:
        Tuple of (L, A, B) channels as separate NumPy arrays
    """
    l, a, b = cv2.split(lab_image)
    return l, a, b

def calculate_greenness(image):
    """
    Calculate "greenness" of an RGB image
    
    As mentioned in Lecture 1B, a simple greenness measure is:
    G - (R+B)/2
    
    Args:
        image: RGB image as NumPy array
        
    Returns:
        Greenness image as NumPy array
    """
    b, g, r = cv2.split(image)
    greenness = g.astype(np.float32) - (r.astype(np.float32) + b.astype(np.float32)) / 2
    
    # Normalize to 0-255 range
    greenness_normalized = np.clip(greenness, 0, None)
    greenness_normalized = (greenness_normalized / np.max(greenness_normalized) * 255).astype(np.uint8)
    
    return greenness_normalized