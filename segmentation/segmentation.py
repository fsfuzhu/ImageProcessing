#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分割模块，提供基于阈值、颜色聚类、边缘检测等多种分割方法
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def threshold_segmentation(image, method='adaptive', block_size=11, constant=2, max_value=255):
    """
    基于阈值的分割
    
    参数:
        image: 输入图像(灰度)
        method: 阈值方法，可选 'global', 'otsu', 'adaptive'
        block_size: 自适应阈值的块大小
        constant: 自适应阈值的常数
        max_value: 二值化后的最大值
        
    返回:
        二值化后的掩码
    """
    # 确保输入是灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 根据指定方法进行阈值分割
    if method == 'global':
        # 全局阈值 - 使用中值作为阈值
        threshold_value = int(np.median(gray))
        _, mask = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
    
    elif method == 'otsu':
        # Otsu自动阈值
        _, mask = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    elif method == 'adaptive':
        # 自适应阈值
        mask = cv2.adaptiveThreshold(
            gray, 
            max_value, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size, 
            constant
        )
    
    else:
        # 默认使用全局阈值
        threshold_value = int(np.median(gray))
        _, mask = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
    
    return mask


def detect_flower_colors(image):
    """
    增强版颜色检测，支持多种花朵颜色
    
    参数:
        image: 输入图像(BGR)
        
    返回:
        花朵颜色区域掩码
    """
    # 转换为不同颜色空间以获取更好的检测效果
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 检测黄色花朵
    # HSV黄色范围1
    lower_yellow1 = np.array([15, 70, 100])
    upper_yellow1 = np.array([45, 255, 255])
    yellow_mask1 = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
    
    # HSV黄色范围2 - 更浅的黄色
    lower_yellow2 = np.array([40, 40, 180])
    upper_yellow2 = np.array([65, 255, 255])
    yellow_mask2 = cv2.inRange(hsv, lower_yellow2, upper_yellow2)
    
    # 在LAB空间中 - b通道对应黄色
    _, _, b_lab = cv2.split(lab)
    _, b_thresh = cv2.threshold(b_lab, 140, 255, cv2.THRESH_BINARY)
    
    # 在RGB空间中 - 高R和G，低B
    r, g, b = cv2.split(rgb)
    yellow_rgb = np.zeros_like(r)
    yellow_rgb[(r > 140) & (g > 140) & (b < r-30)] = 255
    
    # 合并黄色掩码
    yellow_mask = cv2.bitwise_or(yellow_mask1, yellow_mask2)
    yellow_mask = cv2.bitwise_or(yellow_mask, b_thresh)
    yellow_mask = cv2.bitwise_or(yellow_mask, yellow_rgb)
    
    # 检测白色花朵
    # 在HSV空间中 - 高V低S
    h_hsv, s_hsv, v_hsv = cv2.split(hsv)
    white_mask = np.zeros_like(h_hsv)
    white_mask[(v_hsv > 180) & (s_hsv < 80)] = 255
    
    # 在LAB空间中 - 高L
    l_lab, _, _ = cv2.split(lab)
    _, l_thresh = cv2.threshold(l_lab, 180, 255, cv2.THRESH_BINARY)
    
    # 在RGB空间中 - 所有通道都高
    white_rgb = np.zeros_like(r)
    white_rgb[(r > 180) & (g > 180) & (b > 180)] = 255
    
    # 合并白色掩码
    white_mask = cv2.bitwise_or(white_mask, l_thresh)
    white_mask = cv2.bitwise_or(white_mask, white_rgb)
    
    # 将黄色和白色掩码合并到最终掩码
    combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, white_mask)
    
    # 进行形态学操作，改善掩码质量
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return combined_mask


def color_based_segmentation(image, method='kmeans', n_clusters=5, attempts=10, max_iterations=100):
    """
    基于颜色的分割，优化对各种颜色花朵的识别
    
    参数:
        image: 输入图像(BGR)
        method: 分割方法，可选 'kmeans', 'watershed', 'grabcut', 'color_detect'
        n_clusters: K-means聚类数
        attempts: K-means尝试次数
        max_iterations: K-means最大迭代次数
        
    返回:
        分割结果掩码和标记图
    """
    # 首先使用增强版颜色检测
    color_mask = detect_flower_colors(image)
    
    # 重塑图像以用于聚类
    h, w = image.shape[:2]
    reshaped = image.reshape(-1, 3)
    
    if method == 'kmeans':
        # 使用K-means进行颜色聚类
        kmeans = KMeans(
            n_clusters=n_clusters, 
            n_init=attempts, 
            max_iter=max_iterations, 
            random_state=42
        )
        labels = kmeans.fit_predict(reshaped)
        
        # 将标签重塑回原始图像形状
        segmented = labels.reshape(h, w)
        
        # 获取聚类中心
        centers = kmeans.cluster_centers_
        
        # 创建掩码用于存储所有潜在花朵区域
        cluster_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 分析每个聚类，查找可能的花朵颜色
        for i in range(n_clusters):
            center = centers[i]
            b, g, r = center
            
            # 跳过可能是绿色（叶子/茎/草）的聚类
            if g > r+10 and g > b+10:
                continue
                
            # 跳过可能是非常暗的区域（背景）
            if r < 60 and g < 60 and b < 60:
                continue
                
            # 添加这个聚类到掩码中
            cluster_mask[segmented == i] = 255
        
        # 结合颜色检测和聚类结果
        combined_mask = cv2.bitwise_or(color_mask, cluster_mask)
        
        return combined_mask, segmented
    
    elif method == 'color_detect':
        # 直接返回基于颜色检测的结果
        return color_mask, None
    
    elif method == 'watershed':
        # 使用分水岭算法
        # 预处理 - 将图像转为灰度并进行滤波
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 噪声去除
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 确定前景区域
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # 找到未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 应用分水岭算法
        markers = cv2.watershed(image, markers)
        
        # 创建掩码 - 标记为1的区域为背景
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[markers > 1] = 255
        
        # 结合颜色检测结果
        combined_mask = cv2.bitwise_and(mask, color_mask)
        
        # 如果结合后的掩码太小，可能是颜色检测限制太严格，使用原始掩码
        if np.sum(combined_mask) < np.sum(mask) * 0.3:
            combined_mask = mask
        
        return combined_mask, markers
    
    elif method == 'grabcut':
        # 使用GrabCut算法
        # 初始化掩码
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # 使用颜色检测结果初始化GrabCut掩码
        if np.sum(color_mask) > 0:
            # 找到颜色检测掩码的边界矩形
            y_indices, x_indices = np.where(color_mask > 0)
            x_min, y_min = max(0, np.min(x_indices) - 10), max(0, np.min(y_indices) - 10)
            x_max, y_max = min(w-1, np.max(x_indices) + 10), min(h-1, np.max(y_indices) + 10)
            
            rect = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # 将颜色检测的结果标记为可能的前景
            mask[color_mask > 0] = cv2.GC_PR_FGD
            mask[color_mask == 0] = cv2.GC_BGD
        else:
            # 如果颜色检测失败，使用中心区域初始化
            rect = (w//4, h//4, w//2, h//2)
        
        # GrabCut所需的临时数组
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # 应用GrabCut
        try:
            if np.sum(color_mask) > 0:
                cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            else:
                cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # 创建最终掩码
            grabcut_mask = np.zeros_like(mask)
            grabcut_mask[(mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)] = 255
            
            # 如果GrabCut结果很小，可能是初始化问题，回退到颜色检测
            if np.sum(grabcut_mask) < 100 and np.sum(color_mask) > 0:
                return color_mask, mask
            
            return grabcut_mask, mask
        except:
            # 如果GrabCut失败，回退到颜色检测
            if np.sum(color_mask) > 0:
                return color_mask, None
            
            # 如果颜色检测也没有结果，返回空掩码
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return empty_mask, None
    
    else:
        # 默认返回颜色检测结果
        return color_mask, None


def edge_detection(image, method='canny', threshold1=30, threshold2=100, aperture_size=3):
    """
    基于边缘检测的分割
    
    参数:
        image: 输入图像
        method: 边缘检测方法，可选 'canny', 'sobel', 'laplacian'
        threshold1: Canny边缘检测的低阈值
        threshold2: Canny边缘检测的高阈值
        aperture_size: 算子孔径大小
        
    返回:
        边缘检测结果
    """
    # 确保输入是灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method == 'canny':
        # Canny边缘检测
        edges = cv2.Canny(blurred, threshold1, threshold2, apertureSize=aperture_size)
        
        # 膨胀边缘以连接近邻边缘
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 闭合边缘中的缝隙
        closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 查找边缘图像中的轮廓
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 按面积过滤轮廓（移除微小轮廓）
        h, w = gray.shape
        min_area = 0.001 * h * w
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # 创建边缘掩码
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(edge_mask, filtered_contours, -1, 255, -1)
        
        return edge_mask
    
    elif method == 'sobel':
        # Sobel边缘检测
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=aperture_size)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=aperture_size)
        
        # 计算梯度幅值
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # 归一化并转换为8位无符号整型
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 阈值处理
        _, edges = cv2.threshold(magnitude, threshold1, 255, cv2.THRESH_BINARY)
        
        # 应用形态学操作改善边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return edges
    
    elif method == 'laplacian':
        # Laplacian边缘检测
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=aperture_size)
        
        # 取绝对值并归一化
        laplacian = np.abs(laplacian)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 阈值处理
        _, edges = cv2.threshold(laplacian, threshold1, 255, cv2.THRESH_BINARY)
        
        # 应用形态学操作改善边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return edges
    
    else:
        # 默认使用Canny
        edges = cv2.Canny(blurred, threshold1, threshold2, apertureSize=aperture_size)
        
        # 膨胀边缘以连接近邻边缘
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 闭合边缘中的缝隙
        closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return closed_edges


def region_growing(image, seed_selection='auto', threshold=15, connectivity=8):
    """
    区域生长分割
    
    参数:
        image: 输入图像
        seed_selection: 种子点选择方法，可选 'auto', 'center'
        threshold: 区域生长的阈值
        connectivity: 邻域连通性，可选 4 或 8
        
    返回:
        区域生长结果掩码
    """
    # 确保输入是灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 种子点选择
    if seed_selection == 'center':
        # 使用图像中心作为种子点
        seeds = [(h//2, w//2)]
    else:
        # 自动选择种子点 - 使用最亮的区域
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果找到轮廓，使用最大轮廓的中心点
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                seeds = [(cy, cx)]
            else:
                # 如果无法计算中心点，尝试多个种子点
                seeds = [(h//4, w//4), (h//4, 3*w//4), (3*h//4, w//4), (3*h//4, 3*w//4), (h//2, w//2)]
        else:
            # 如果没有找到轮廓，尝试多个种子点
            seeds = [(h//4, w//4), (h//4, 3*w//4), (3*h//4, w//4), (3*h//4, 3*w//4), (h//2, w//2)]
    
    # 区域生长
    for seed_y, seed_x in seeds:
        # 边界检查
        if 0 <= seed_y < h and 0 <= seed_x < w:
            # 创建标记掩码
            visited = np.zeros((h, w), dtype=np.uint8)
            
            # 初始化队列并添加种子点
            queue = [(seed_y, seed_x)]
            seed_value = int(gray[seed_y, seed_x])
            
            while queue:
                y, x = queue.pop(0)
                
                # 如果该点未访问过且满足阈值条件
                if visited[y, x] == 0 and abs(int(gray[y, x]) - seed_value) <= threshold:
                    # 标记为已访问
                    visited[y, x] = 1
                    mask[y, x] = 255
                    
                    # 添加邻域点到队列
                    if connectivity == 4:
                        neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
                    else:  # connectivity == 8
                        neighbors = [
                            (y-1, x), (y+1, x), (y, x-1), (y, x+1),
                            (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)
                        ]
                    
                    for ny, nx in neighbors:
                        if 0 <= ny < h and 0 <= nx < w and visited[ny, nx] == 0:
                            queue.append((ny, nx))
    
    return mask


def segment_flower(image, original_image=None, config=None):
    """
    执行花朵分割的综合函数
    
    参数:
        image: 预处理后的输入图像（可能已转换颜色空间）
        original_image: 原始图像，用于某些需要原始数据的算法
        config: 分割配置参数
        
    返回:
        分割结果，掩码集合和特征
    """
    if original_image is None:
        original_image = image.copy()
    
    if config is None:
        config = {
            'threshold': {'enabled': True, 'method': 'adaptive'},
            'color': {'enabled': True, 'method': 'kmeans'},
            'edge': {'enabled': True, 'method': 'canny'},
            'region': {'enabled': True}
        }
    
    # 转换为灰度图（如果需要）
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 初始化结果
    segmented = original_image.copy()
    masks = {}
    features = {}
    
    # 阈值分割
    if config['threshold']['enabled']:
        threshold_mask = threshold_segmentation(
            gray, 
            method=config['threshold']['method'],
            block_size=config['threshold'].get('block_size', 11),
            constant=config['threshold'].get('constant', 2),
            max_value=config['threshold'].get('max_value', 255)
        )
        masks['threshold'] = threshold_mask
    
    # 颜色分割
    if config['color']['enabled']:
        color_mask, color_segments = color_based_segmentation(
            original_image, 
            method=config['color']['method'],
            n_clusters=config['color'].get('n_clusters', 5),
            attempts=config['color'].get('attempts', 10),
            max_iterations=config['color'].get('max_iterations', 100)
        )
        masks['color'] = color_mask
        features['color_segments'] = color_segments
        
        # 增加对黄色和白色花朵的特殊处理
        if np.sum(color_mask) > 0:
            # 单独保存黄色掩码
            yellow_mask = detect_flower_colors(original_image)
            masks['yellow'] = yellow_mask
    
    # 边缘检测
    if config['edge']['enabled']:
        edge_mask = edge_detection(
            gray, 
            method=config['edge']['method'],
            threshold1=config['edge'].get('threshold1', 30),
            threshold2=config['edge'].get('threshold2', 100),
            aperture_size=config['edge'].get('aperture_size', 3)
        )
        masks['edge'] = edge_mask
    
    # 区域生长
    if config['region']['enabled']:
        region_mask = region_growing(
            gray, 
            seed_selection=config['region'].get('seed_selection', 'auto'),
            threshold=config['region'].get('threshold', 15),
            connectivity=config['region'].get('connectivity', 8)
        )
        masks['region'] = region_mask
    
    # 最终分割结果
    # 使用加权投票机制，颜色分割有更高权重
    final_mask = np.zeros_like(gray, dtype=np.uint8)
    
    if masks:
        # 初始化投票矩阵
        votes = np.zeros_like(gray, dtype=np.float32)
        
        # 颜色分割有更高权重
        color_weight = 1.5
        if 'color' in masks:
            votes = votes + (masks['color'] / 255.0) * color_weight
        
        # 其他方法投票
        for method, mask in masks.items():
            if method != 'color' and method != 'yellow':
                votes = votes + (mask / 255.0)
        
        # 如果超过一半方法认为是前景，则标记为前景
        threshold = (len(masks) - 1 + color_weight) / 2.0
        final_mask[votes >= threshold] = 255
        
        # 如果颜色分割生成的掩码不为空，且投票结果较小，可能是投票阈值太高
        # 在这种情况下，更信任颜色分割的结果
        if 'color' in masks and np.sum(final_mask) < np.sum(masks['color']) * 0.5:
            final_mask = masks['color']
    
    # 应用最终掩码到原始图像
    segmented = cv2.bitwise_and(original_image, original_image, mask=final_mask)
    
    # 添加最终掩码到结果集
    masks['final'] = final_mask
    
    return segmented, masks, features