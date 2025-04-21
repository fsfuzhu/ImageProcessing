#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP2032 Coursework - Modified Pipeline with Purple Flower and Bokeh Background Detection
Flower segmentation pipeline with smooth edge preservation and special handling for special cases
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
        self.min_flower_size_ratio = 0.1  # Adjusted minimum flower size ratio for inversion check

        # Internal flag for purple flower case
        self._is_purple_flower_case = False

    def _detect_bokeh_background(self, image):
        """
        Detect if the image has a bokeh (blurred) background

        Args:
            image: Input RGB image (NumPy array)

        Returns:
            Boolean indicating if the image has a bokeh background
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Divide image into center and edges
        center_size = min(w, h) // 3
        center_y, center_x = h // 2, w // 2

        # Extract center region
        center_region = gray[center_y-center_size:center_y+center_size,
                             center_x-center_size:center_x+center_size]

        # Extract edge regions
        border_thickness = min(w, h) // 8
        border_mask = np.ones_like(gray, dtype=bool)
        border_mask[border_thickness:-border_thickness, border_thickness:-border_thickness] = False
        border_region = gray[border_mask]

        # Calculate focus measure (variance of Laplacian)
        center_focus = cv2.Laplacian(center_region, cv2.CV_64F).var()
        border_focus = cv2.Laplacian(border_region, cv2.CV_64F).var()

        # If center is significantly more in focus than borders, it's likely bokeh
        return center_focus > border_focus * 2 and border_focus < 100

    def _process_bokeh_flower(self, image):
        """
        Special processing for flowers with blurred bokeh backgrounds
        Uses focus detection to separate sharp foreground from blurred background

        Args:
            image: Input RGB image (NumPy array)

        Returns:
            binary_mask: Mask isolating the sharp flower from blurred background
        """
        # Convert to grayscale for focus detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate variance of Laplacian as a measure of focus
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.zeros_like(gray, dtype=np.float32)

        # Calculate local variance in a sliding window
        window_size = 15
        for y in range(window_size//2, gray.shape[0] - window_size//2):
            for x in range(window_size//2, gray.shape[1] - window_size//2):
                window = laplacian[y-window_size//2:y+window_size//2+1,
                                  x-window_size//2:x+window_size//2+1]
                laplacian_var[y, x] = window.var()

        # Normalize variance map
        laplacian_var = cv2.normalize(laplacian_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Blur to smooth out noise
        laplacian_var = cv2.GaussianBlur(laplacian_var, (15, 15), 0)

        # Apply OTSU thresholding to separate focused from blurred areas
        _, focus_mask = cv2.threshold(laplacian_var, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use color information to enhance the mask
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create masks for purple and yellow (common pansy colors)
        purple_lower = np.array([125, 40, 40])
        purple_upper = np.array([160, 255, 255])
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # Also detect light/white parts
        white_lower = np.array([0, 0, 150])
        white_upper = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        # Combine all masks
        color_mask = cv2.bitwise_or(purple_mask, yellow_mask)
        color_mask = cv2.bitwise_or(color_mask, white_mask)

        # Combine focus and color masks
        combined_mask = cv2.bitwise_or(focus_mask, color_mask)

        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find the largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask)

        if num_labels > 1:
            max_area = 0
            max_label = 0

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > max_area:
                    max_area = area
                    max_label = i

            final_mask = np.zeros_like(combined_mask)
            final_mask[labels == max_label] = 255
        else:
            final_mask = combined_mask

        return final_mask

    def _detect_purple_flower_green_background(self, image):
        """
        Detect if the image contains a purple flower with green background

        Args:
            image: Input RGB image (NumPy array)

        Returns:
            Boolean indicating if this is a purple flower with green background
        """
        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        # Define purple color ranges in HSV
        purple_lower_1 = np.array([130, 40, 40])  # Lower purple
        purple_upper_1 = np.array([160, 255, 255])  # Upper purple
        # Note: HSV Hue wraps around. 270-320 degrees in H would be handled by modulo 180 in OpenCV's H channel (0-179).
        # 270 / 2 = 135, 320 / 2 = 160. So the ranges might overlap or need careful definition.
        # Let's adjust the second range based on common purple/violet HSV values.
        purple_lower_2 = np.array([150, 40, 40]) # More violet/magenta
        purple_upper_2 = np.array([179, 255, 255])

        # Create purple mask
        purple_mask_1 = cv2.inRange(hsv_image, purple_lower_1, purple_upper_1)
        purple_mask_2 = cv2.inRange(hsv_image, purple_lower_2, purple_upper_2)
        purple_mask = cv2.bitwise_or(purple_mask_1, purple_mask_2)

        # Define green color range in HSV
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])

        # Create green mask
        green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

        # Calculate proportions
        total_pixels = image.shape[0] * image.shape[1]
        purple_pixels = np.sum(purple_mask > 0)
        green_pixels = np.sum(green_mask > 0)

        purple_ratio = purple_pixels / total_pixels
        green_ratio = green_pixels / total_pixels

        # Check if this is a purple flower with green background
        is_purple_flower = purple_ratio > 0.05 and purple_ratio < 0.4  # Significant purple, but not overwhelming
        has_green_background = green_ratio > 0.1  # Significant green background

        return is_purple_flower and has_green_background

    def _process_purple_flower(self, image):
        """
        Special processing for purple flowers with green backgrounds

        Args:
            image: Input RGB image (NumPy array)

        Returns:
            binary_mask: Mask isolating the purple flower
        """
        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define purple color ranges in HSV - broader ranges for better coverage
        # Range 1: Main purple
        purple_lower_1 = np.array([125, 40, 30])
        purple_upper_1 = np.array([165, 255, 255])

        # Range 2: Pinkish purple
        purple_lower_2 = np.array([140, 20, 50])
        purple_upper_2 = np.array([180, 255, 255])

        # Range 3: Bluish purple
        purple_lower_3 = np.array([110, 40, 40])
        purple_upper_3 = np.array([135, 255, 255])

        # Create masks for each purple range
        purple_mask_1 = cv2.inRange(hsv_image, purple_lower_1, purple_upper_1)
        purple_mask_2 = cv2.inRange(hsv_image, purple_lower_2, purple_upper_2)
        purple_mask_3 = cv2.inRange(hsv_image, purple_lower_3, purple_upper_3)

        # Combine purple masks
        purple_mask = cv2.bitwise_or(purple_mask_1, purple_mask_2)
        purple_mask = cv2.bitwise_or(purple_mask, purple_mask_3)

        # Yellow center detection for pansy flowers
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

        # Include yellow center if it's surrounded by purple
        kernel = np.ones((5,5), np.uint8)
        purple_dilated = cv2.dilate(purple_mask, kernel, iterations=2)
        yellow_center = cv2.bitwise_and(yellow_mask, yellow_mask, mask=purple_dilated)

        # Combine purple and yellow center
        combined_mask = cv2.bitwise_or(purple_mask, yellow_center)

        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        # Remove noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Fill holes
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Find the largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)

        if num_labels > 1:
            # Find the largest component (excluding background)
            max_area = 0
            max_label = 1
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > max_area:
                    max_area = area
                    max_label = i

            # Extract the largest component
            final_mask = np.zeros_like(combined_mask)
            if max_label > 0:  # Ensure a component was found
                final_mask[labels == max_label] = 255
            else:
                # If no significant component found, return the combined mask as is
                final_mask = combined_mask
        else:
            # If no components found, return the combined mask as is
            final_mask = combined_mask

        return final_mask


    def _detect_and_fix_mask_inversion(self, mask, original_image):
        """
        Detect if the mask is inverted (flower is black, background is white) and fix it
        with a more reasonable judgment.

        Args:
            mask: Binary mask
            original_image: Original RGB image (used for potential color checks, though simplified now)

        Returns:
            Fixed mask with flower as white and background as black
        """
        # Calculate mask proportions
        total_pixels = mask.shape[0] * mask.shape[1]
        white_pixels = np.sum(mask > 127)
        white_ratio = white_pixels / total_pixels

        # Define a threshold for the minimum expected size of the flower in the mask
        # If the white area is smaller than this, it's likely the mask is inverted
        min_white_ratio_threshold = self.min_flower_size_ratio

        # Invert the mask if the white area is too small (suggesting the flower is black)
        needs_inversion = False
        if white_ratio < min_white_ratio_threshold:
            print(f"White area ratio ({white_ratio:.4f}) is below threshold ({min_white_ratio_threshold}), considering inversion...")

            # Further check: is there a large black area in the center?
            h, w = mask.shape[:2]
            center_x, center_y = w // 2, h // 2
            center_size = min(w, h) // 4 # Use a reasonable center size
            center_region = mask[
                center_y - center_size:center_y + center_size,
                center_x - center_size:center_x + center_size
            ]
            black_pixels_in_center = np.sum(center_region < 127)
            center_total_pixels = center_region.size
            black_ratio_in_center = black_pixels_in_center / center_total_pixels if center_total_pixels > 0 else 0

            # If the white area is small AND there's a significant black area in the center,
            # it strongly suggests the flower (expected in the center) is black.
            if black_ratio_in_center > 0.5: # Threshold for significant black in center
                 needs_inversion = True
                 print("Significant black area detected in the center. Inverting mask.")
            else:
                 print("Not enough black in the center to confirm inversion. Mask not inverted.")


        # Invert the mask if needed
        if needs_inversion:
             return cv2.bitwise_not(mask)
        else:
             return mask


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

        # Check for special cases
        has_bokeh = self._detect_bokeh_background(original_image)
        is_purple_flower = self._detect_purple_flower_green_background(original_image)
        self._is_purple_flower_case = is_purple_flower  # Store for use in mask inversion detection

        if has_bokeh:
            # Use special processing for bokeh backgrounds
            print("Detected bokeh background, using focus-based processing...")
            watershed_mask = self._process_bokeh_flower(original_image)

            # Save intermediate mask
            intermediate_results["special_bokeh_mask"] = watershed_mask

        elif is_purple_flower:
            # Use special processing for purple flowers
            print("Detected purple flower with green background, using special HSV processing...")
            watershed_mask = self._process_purple_flower(original_image)

            # Save intermediate mask
            intermediate_results["special_purple_mask"] = watershed_mask

        else:
            # Normal processing pipeline
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

        # Check and fix mask inversion if needed
        watershed_mask = self._detect_and_fix_mask_inversion(watershed_mask, original_image)
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

        # Corrected line:
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
            min_area = total_pixels * self.min_flower_size_ratio # Use the class's min_flower_size_ratio

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                # Only consider components larger than a minimum size ratio
                if area > max_area and area >= min_area:
                    max_area = area
                    max_label = i

            # Extract largest component
            largest_component = np.zeros_like(binary_mask)
            if max_label > 0:  # Ensure a significant component was found
                largest_component[labels == max_label] = 255
            else:
                # If no component larger than the minimum size found, return an empty mask
                print("No component larger than minimum size found after filling holes.")
                largest_component = np.zeros_like(binary_mask) # Return an empty mask

        else:
            # If no components found (only background), return an empty mask
            print("No connected components found after filling holes.")
            largest_component = np.zeros_like(binary_mask)

        return largest_component

    def _create_feathered_mask(self, binary_mask):
        """
        Create a feathered mask with smooth edges for natural-looking segmentation
        by directly blurring the binary mask.

        Args:
            binary_mask: Binary mask of the object (expected to be 0 or 255)

        Returns:
            Feathered mask with smooth transitions at edges (values between 0 and 255)
        """
        # Check if the binary mask is empty. If so, return an empty feathered mask.
        if np.sum(binary_mask) == 0:
             print("Binary mask is empty, returning empty feathered mask.")
             return np.zeros_like(binary_mask, dtype=np.uint8)

        # Apply Gaussian blur directly to the binary mask to create feathering
        # The kernel size determines the width of the feathered edge.
        # Ensure the kernel size is odd.
        blur_size = self.edge_blur_size
        if blur_size % 2 == 0:
             blur_size += 1
             if blur_size < 3: blur_size = 3 # Ensure a minimum kernel size

        # Apply Gaussian blur
        # Using the same kernel size for both width and height
        feathered_mask = cv2.GaussianBlur(binary_mask, (blur_size, blur_size), 0)

        # The blurred result will have values between 0 and 255 naturally.
        # We can optionally re-normalize, but usually, direct blur is sufficient
        # if the input is 0 or 255. Let's normalize to be safe.
        feathered_mask = cv2.normalize(feathered_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return feathered_mask