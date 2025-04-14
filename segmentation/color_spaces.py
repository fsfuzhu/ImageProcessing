#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - General Version
Color Space Conversion Utilities
Based on Lecture 1B - Digital Images and Point Processing
"""

import cv2
import numpy as np

def convert_to_hsv(image):
    """
    Convert an RGB image to HSV color space
    
    HSV separates color (Hue) from intensity (Value), making it less sensitive 
    to lighting changes, which can be beneficial for segmentation.
    
    Args:
        image: RGB image as a NumPy array
        
    Returns:
        HSV image as a NumPy array
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def convert_to_lab(image):
    """
    Convert an RGB image to LAB color space
    
    LAB separates lightness (L) from color information (A and B channels), 
    making it robust to brightness variations when differentiating colors.
    
    Args:
        image: RGB image as a NumPy array
        
    Returns:
        LAB image as a NumPy array
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

def convert_to_grayscale(image):
    """
    Convert an RGB image to grayscale using a weighted formula
    
    As discussed in Lecture 1B, we can convert RGB to grayscale using:
    I = 0.30R + 0.59G + 0.11B (weights based on human eye sensitivity)
    
    Args:
        image: RGB image as a NumPy array
        
    Returns:
        Grayscale image as a NumPy array
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def extract_hsv_channels(hsv_image):
    """
    Extract individual channels from an HSV image
    
    Args:
        hsv_image: HSV image as a NumPy array
        
    Returns:
        Tuple of (H, S, V) channels as separate NumPy arrays
    """
    h, s, v = cv2.split(hsv_image)
    return h, s, v

def extract_lab_channels(lab_image):
    """
    Extract individual channels from a LAB image
    
    Args:
        lab_image: LAB image as a NumPy array
        
    Returns:
        Tuple of (L, A, B) channels as separate NumPy arrays
    """
    l, a, b = cv2.split(lab_image)
    return l, a, b

def calculate_greenness(image):
    """
    Calculate the "greenness" of an RGB image
    
    As discussed in Lecture 1B, a simple greenness measure is:
    G - (R+B)/2
    
    Args:
        image: RGB image as a NumPy array
        
    Returns:
        Greenness image as a NumPy array
    """
    b, g, r = cv2.split(image)
    greenness = g.astype(np.float32) - (r.astype(np.float32) + b.astype(np.float32)) / 2
    
    # Normalize to 0-255 range
    greenness_normalized = np.clip(greenness, 0, None)
    if np.max(greenness_normalized) > 0:
        greenness_normalized = (greenness_normalized / np.max(greenness_normalized) * 255).astype(np.uint8)
    else:
        greenness_normalized = np.zeros_like(greenness, dtype=np.uint8)
    
    return greenness_normalized