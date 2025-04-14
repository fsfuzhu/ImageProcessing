#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - General Version
Noise Reduction Implementations
Based on Lecture 3A (Linear Filters) and 3B (Non-linear Filters)
"""

import cv2
import numpy as np

def apply_mean_filter(image, kernel_size=(3, 3)):
    """
    Apply a mean filter for noise reduction
    
    Mean filtering is a simple spatial filter that replaces each pixel value
    with the mean value of its neighborhood. It's effective for Gaussian noise 
    but blurs edges.
    
    Args:
        image: Input image as a NumPy array
        kernel_size: Size of the filter kernel as a tuple (width, height)
        
    Returns:
        Filtered image as a NumPy array
    """
    return cv2.blur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=0):
    """
    Apply a Gaussian filter for noise reduction
    
    Gaussian filtering uses a weighted average, where pixels closer to the 
    center get higher weights. It preserves edges better than mean filtering.
    
    Args:
        image: Input image as a NumPy array
        kernel_size: Size of the filter kernel as a tuple (width, height)
        sigma: Standard deviation of the Gaussian. If 0, it's computed from kernel size.
        
    Returns:
        Filtered image as a NumPy array
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_median_filter(image, kernel_size=3):
    """
    Apply a median filter for noise reduction
    
    Median filtering replaces each pixel with the median value of its neighborhood.
    It's particularly effective against salt-and-pepper noise while preserving edges.
    
    Args:
        image: Input image as a NumPy array
        kernel_size: Size of the square filter kernel (must be odd)
        
    Returns:
        Filtered image as a NumPy array
    """
    return cv2.medianBlur(image, kernel_size)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply a bilateral filter for edge-preserving smoothing
    
    Bilateral filtering considers both spatial closeness and color similarity,
    effectively preserving edges while smoothing flat regions.
    
    Args:
        image: Input image as a NumPy array
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Filtered image as a NumPy array
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_anisotropic_diffusion(image, iterations=5, k=25, gamma=0.1):
    """
    Apply an anisotropic diffusion filter (Perona-Malik)
    
    This is an implementation of the anisotropic diffusion filter based on Lecture 3B.
    It smooths the image while preserving edges.
    
    Args:
        image: Input image as NumPy array
        iterations: Number of iterations
        k: Diffusion constant (controls sensitivity to edges)
        gamma: Controls the diffusion speed
        
    Returns:
        Filtered image as NumPy array
    """
    # Convert to float32 for processing
    img_float = image.astype(np.float32)
    
    # Create output image
    out = img_float.copy()
    
    # Define delta for 4-neighbor diffusion
    delta = [
        (0, 1),   # Right
        (0, -1),  # Left
        (1, 0),   # Down
        (-1, 0)   # Up
    ]
    
    # Apply diffusion iterations
    for _ in range(iterations):
        # Calculate diffusion for each direction
        diffusion = np.zeros_like(out)
        
        for dx, dy in delta:
            # Calculate shifted array
            shifted = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            
            # Calculate difference between current and shifted image
            diff = shifted - out
            
            # Apply diffusion function (Perona-Malik)
            # g(x) = 1 / (1 + (x/k)^2)
            g = 1.0 / (1.0 + (diff / k) ** 2)
            
            # Accumulate diffusion
            diffusion += g * diff
        
        # Update the image
        out += gamma * diffusion
    
    # Convert back to uint8
    return np.clip(out, 0, 255).astype(np.uint8)