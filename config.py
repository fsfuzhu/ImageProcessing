#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件，包含所有可调整的参数
"""

class Config:
    """配置类，包含所有分割流程的参数"""
    
    def __init__(self):
        """初始化配置参数"""
        
        # 预处理参数
        self.preprocessing = {
            # 图像尺寸调整
            'resize': {
                'enabled': True,
                'width': 256,
                'height': 256
            },
            # 去噪
            'denoise': {
                'enabled': True,
                'method': 'gaussian',  # 'gaussian', 'median', 'bilateral'
                'kernel_size': 5,
                'sigma': 1.0
            },
            # 对比度增强
            'contrast': {
                'enabled': True,
                'method': 'clahe',  # 'clahe', 'histogram_equalization'
                'clip_limit': 2.0,
                'tile_grid_size': (8, 8)
            },
            # 图像锐化
            'sharpen': {
                'enabled': False,
                'kernel_size': 3,
                'sigma': 1.0,
                'amount': 1.5,
                'threshold': 0
            }
        }
        
        # 颜色空间转换参数
        self.color_space = {
            'space': 'hsv',  # 'rgb', 'hsv', 'lab', 'ycrcb'
            'channels': [0, 1, 2]  # 用于分割的通道索引
        }
        
        # 分割参数
        self.segmentation = {
            # 阈值分割
            'threshold': {
                'enabled': True,
                'method': 'adaptive',  # 'global', 'otsu', 'adaptive'
                'block_size': 11,
                'constant': 2,
                'max_value': 255
            },
            # 颜色分割
            'color': {
                'enabled': True,
                'method': 'kmeans',  # 'kmeans', 'watershed', 'grabcut'
                'n_clusters': 5,
                'attempts': 10,
                'max_iterations': 100
            },
            # 边缘检测
            'edge': {
                'enabled': True,
                'method': 'canny',  # 'canny', 'sobel', 'laplacian'
                'threshold1': 50,
                'threshold2': 150,
                'aperture_size': 3
            },
            # 区域生长
            'region': {
                'enabled': False,
                'seed_selection': 'auto',  # 'auto', 'center'
                'threshold': 10,
                'connectivity': 8
            }
        }
        
        # 后处理参数
        self.postprocessing = {
            # 形态学操作
            'morphology': {
                'enabled': True,
                'opening': {
                    'enabled': True,
                    'kernel_size': 3,
                    'iterations': 1
                },
                'closing': {
                    'enabled': True,
                    'kernel_size': 3,
                    'iterations': 2
                },
                'dilation': {
                    'enabled': True,
                    'kernel_size': 3,
                    'iterations': 1
                },
                'erosion': {
                    'enabled': False,
                    'kernel_size': 3,
                    'iterations': 1
                }
            },
            # 连通区域分析
            'connected_components': {
                'enabled': True,
                'min_area_ratio': 0.01,  # 相对于图像大小的最小连通区域比例
                'max_num_components': 3  # 保留的最大连通区域数量
            },
            # 边界平滑
            'contour_smoothing': {
                'enabled': True,
                'method': 'gaussian',  # 'gaussian', 'median'
                'kernel_size': 5
            },
            # 孔洞填充
            'hole_filling': {
                'enabled': True,
                'min_hole_area': 50  # 填充的最小孔洞面积
            }
        }