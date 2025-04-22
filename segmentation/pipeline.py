#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - Modified Pipeline
Flower segmentation pipeline with smooth edge preservation
"""

import cv2
import numpy as np
from .color_spaces import convert_to_lab, extract_lab_channels
from .noise_reduction import apply_gaussian_filter, apply_bilateral_filter
from .thresholding import adaptive_threshold, otsu_threshold, multi_level_threshold
from .morphology import fill_holes

class FlowerSegmentationPipeline:
    """
    Modified flower segmentation pipeline with edge feathering for natural-looking results
    """
    
    def __init__(self):
        """Initialize segmentation pipeline with parameters"""
        # Enhancement and filtering parameters
        self.gaussian_kernel = (3, 3)  # Gaussian filter kernel size
        self.bilateral_d = 60  # Bilateral filter diameter
        self.bilateral_sigma = 50  # Bilateral filter sigma parameter
        
        # Threshold parameters
        self.adaptive_block_size = 15  # Adaptive threshold block size
        self.adaptive_c = 1  # Adaptive threshold constant
        
        # Edge feathering parameters
        self.edge_blur_size = 11  # Size of Gaussian blur for edge feathering
        self.feather_amount = 12  # Amount of feathering to apply
        
        # Watershed parameters
        self.min_flower_size_ratio = 0.35  # Minimum flower size ratio
    
    def process(self, image):
        """
        Process image through the segmentation pipeline with smooth edge preservation
        
        Args:
            image: Input RGB image (NumPy array)
            
        Returns:
            segmented_image: Image with segmented flower and smooth edges
            intermediate_results: Dictionary of intermediate processing steps
        """
        # Store intermediate results for visualization
        intermediate_results = {}
        
        # Step 1: Use original image without scaling
        original_image = image.copy()
        intermediate_results["1_original"] = original_image
        
        # Step 2: Convert to LAB color space
        lab_image = convert_to_lab(original_image)
        l, a, b = extract_lab_channels(lab_image)
        
        # Save LAB channels
        intermediate_results["2a_lab_l_channel"] = l
        intermediate_results["2b_lab_a_channel"] = a
        intermediate_results["2c_lab_b_channel"] = b
        
        # Step 3: Enhance contrast using histogram equalization
        enhanced_l = cv2.equalizeHist(l)
        intermediate_results["3a_enhanced_l"] = enhanced_l
        
        # Calculate gradient information
        grad_l = self._calculate_gradient(l)
        grad_a = self._calculate_gradient(a)
        grad_b = self._calculate_gradient(b)
        
        # Combine gradient information
        combined_gradient = cv2.addWeighted(grad_l, 0.5, cv2.addWeighted(grad_a, 0.5, grad_b, 0.5, 0), 0.5, 0)
        intermediate_results["3b_combined_gradient"] = combined_gradient
        
        # Step 4: Apply filters for noise reduction
        gaussian_filtered = apply_gaussian_filter(enhanced_l, self.gaussian_kernel)
        intermediate_results["4a_gaussian_filtered"] = gaussian_filtered
        
        bilateral_filtered = apply_bilateral_filter(
            gaussian_filtered, 
            d=self.bilateral_d, 
            sigma_color=self.bilateral_sigma, 
            sigma_space=self.bilateral_sigma
        )
        intermediate_results["4b_bilateral_filtered"] = bilateral_filtered
        
        # Step 5: Multi-level thresholding
        adaptive_thresh = adaptive_threshold(
            bilateral_filtered, 
            block_size=self.adaptive_block_size, 
            c=self.adaptive_c
        )
        intermediate_results["5a_adaptive_threshold"] = adaptive_thresh
        
        otsu_thresh = otsu_threshold(bilateral_filtered)
        intermediate_results["5b_otsu_threshold"] = otsu_thresh
        
        multi_thresh = multi_level_threshold(bilateral_filtered, num_classes=3)
        multi_thresh_binary = cv2.threshold(multi_thresh, 127, 255, cv2.THRESH_BINARY)[1]
        intermediate_results["5c_multi_threshold"] = multi_thresh_binary
        
        # Combine threshold results
        combined_thresh = cv2.bitwise_or(
            adaptive_thresh, 
            cv2.bitwise_or(otsu_thresh, multi_thresh_binary)
        )
        intermediate_results["5d_combined_threshold"] = combined_thresh
        
        # Step 6: Apply watershed segmentation
        watershed_mask = self._apply_watershed_segmentation(original_image, combined_thresh)
        intermediate_results["6a_watershed_segmentation"] = watershed_mask
        
        # Fill holes
        filled_mask = fill_holes(watershed_mask)
        intermediate_results["6b_filled_holes"] = filled_mask
        
        # Extract largest component
        largest_component = self._extract_largest_component(filled_mask)
        intermediate_results["6c_largest_component"] = largest_component
        
        # Step 7: Create feathered edges for smooth transitions
        feathered_mask = self._create_feathered_mask(largest_component)
        intermediate_results["7a_feathered_mask"] = feathered_mask
        
        # Step 8: Apply the feathered mask to the original image
        # Instead of a binary segmentation, use the feathered mask for a gradual transition
        # Create a 3-channel mask by duplicating the feathered mask for RGB
        h, w = feathered_mask.shape[:2]
        alpha_mask = np.zeros((h, w, 3), dtype=np.float32)
        alpha_mask[:, :, 0] = feathered_mask / 255.0
        alpha_mask[:, :, 1] = feathered_mask / 255.0
        alpha_mask[:, :, 2] = feathered_mask / 255.0
        
        # Apply the alpha mask to create the final segmented image with smooth edges
        segmented_flower = np.zeros_like(original_image)
        for c in range(3):
            segmented_flower[:, :, c] = original_image[:, :, c] * alpha_mask[:, :, c]
        
        intermediate_results["8_final_segmented"] = segmented_flower
        
        return segmented_flower, intermediate_results
    
    def _calculate_gradient(self, image):
        """Calculate image gradient and return normalized gradient magnitude"""
        # Use Sobel operator to calculate gradient
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = cv2.magnitude(sobelx, sobely)
        
        # Normalize to 0-255 range
        normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized.astype(np.uint8)
    
    def _apply_watershed_segmentation(self, image, binary_mask):
        """Apply watershed algorithm for segmentation"""
        # Ensure binary mask
        _, sure_fg = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find sure foreground regions using distance transform
        dist_transform = cv2.distanceTransform(sure_fg, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Find sure background regions
        sure_bg = cv2.dilate(binary_mask, np.ones((3,3), np.uint8), iterations=3)
        
        # Calculate unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Label markers
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so background isn't 0
        markers = markers + 1
        
        # Mark unknown region as 0
        markers[unknown == 255] = 0
        
        # Apply watershed algorithm
        color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 2 else image
        markers = cv2.watershed(color_image, markers.astype(np.int32))
        
        # Create watershed result mask
        watershed_mask = np.zeros_like(binary_mask)
        watershed_mask[markers > 1] = 255
        
        return watershed_mask
    
    def _extract_largest_component(self, binary_mask):
        """Extract the largest connected component from binary mask"""
        # Perform connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # Skip background (label 0)
        if num_labels > 1:
            # Find largest component by area (excluding background)
            max_area = 0
            max_label = 0
            
            # Total image pixels
            total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
            min_area = total_pixels * self.min_flower_size_ratio
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > max_area and area > min_area:
                    max_area = area
                    max_label = i
            
            # Extract largest component
            largest_component = np.zeros_like(binary_mask)
            if max_label > 0:  # Ensure a component was found
                largest_component[labels == max_label] = 255
            else:
                # If no component found, use original mask
                largest_component = binary_mask
        else:
            # If no components found, use original mask
            largest_component = binary_mask
            
        return largest_component
    
    def _create_feathered_mask(self, binary_mask):
        """
        Create a feathered mask with smooth edges for natural-looking segmentation
        
        Args:
            binary_mask: Binary mask of the object
            
        Returns:
            Feathered mask with smooth transitions at edges
        """
        # First, apply a slight dilation to ensure we capture all of the flower
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        
        # Apply distance transform to find distances to the nearest background pixel
        dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 3)
        
        # Normalize the distance transform to 0-255 range
        dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        dist_norm = dist_norm.astype(np.uint8)
        
        # Apply Gaussian blur to create feathering effect
        feathered_mask = cv2.GaussianBlur(dist_norm, (self.edge_blur_size, self.edge_blur_size), 0)
        
        # Create a gradient transition at the edges using the feather_amount parameter
        _, binary_feathered = cv2.threshold(feathered_mask, 1, 255, cv2.THRESH_BINARY)
        
        # Dilate the binary mask to define the feathering region
        feather_region = cv2.dilate(binary_feathered, 
                                   np.ones((self.feather_amount*2+1, self.feather_amount*2+1), np.uint8), 
                                   iterations=1)
        feather_region = cv2.subtract(feather_region, binary_feathered)
        
        # Apply the feathered mask
        result = binary_feathered.copy()
        result[feather_region > 0] = feathered_mask[feather_region > 0]
        
        return result