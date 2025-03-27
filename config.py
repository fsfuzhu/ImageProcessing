#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件，包含所有可调整的参数 - 针对多种颜色花朵优化
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
                'method': 'bilateral',  # 'gaussian', 'median', 'bilateral'
                'kernel_size': 5,
                'sigma': 1.5
            },
            # 对比度增强
            'contrast': {
                'enabled': True,
                'method': 'clahe',  # 'clahe', 'histogram_equalization'
                'clip_limit': 3.0,
                'tile_grid_size': (8, 8)
            },
            # 图像锐化
            'sharpen': {
                'enabled': True,
                'kernel_size': 3,
                'sigma': 1.0,
                'amount': 1.8,
                'threshold': 5
            }
        }
        
        # 颜色空间转换参数 - 通用优化
        self.color_space = {
            'space': 'hsv',  # 'rgb', 'hsv', 'lab', 'ycrcb'
            'channels': [0, 1, 2]  # 用于分割的通道索引
        }
        
        # 分割参数
        self.segmentation = {
            # 阈值分割
            'threshold': {
                'enabled': True,
                'method': 'otsu',  # 'global', 'otsu', 'adaptive'
                'block_size': 11,
                'constant': 2,
                'max_value': 255
            },
            # 颜色分割 - 优化对各种颜色花朵的识别
            'color': {
                'enabled': True,
                'method': 'kmeans',  # 'kmeans', 'watershed', 'grabcut', 'color_detect'
                'n_clusters': 5,  # 增加聚类数，识别更多颜色
                'attempts': 15,  # 增加尝试次数提高稳定性
                'max_iterations': 150  # 增加迭代次数
            },
            # 边缘检测
            'edge': {
                'enabled': True,
                'method': 'canny',  # 'canny', 'sobel', 'laplacian'
                'threshold1': 30,  # 降低阈值捕获更多边缘
                'threshold2': 100,
                'aperture_size': 3
            },
            # 区域生长
            'region': {
                'enabled': True,
                'seed_selection': 'auto',  # 'auto', 'center'
                'threshold': 15,  # 增加阈值以包含更多相似颜色
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
                    'kernel_size': 7,  # 增大核大小填补更多空隙
                    'iterations': 3    # 增加迭代次数
                },
                'dilation': {
                    'enabled': True,
                    'kernel_size': 5,  # 增大核大小
                    'iterations': 2    # 增加迭代次数
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
                'min_area_ratio': 0.001,  # 减小该值以保留更小的区域
                'max_num_components': 5   # 增加保留区域数量
            },
            # 边界平滑
            'contour_smoothing': {
                'enabled': True,
                'method': 'combined',  # 'gaussian', 'median', 'combined', 'contour'
                'kernel_size': 5  # 减小平滑核，保留更多细节
            },
            # 孔洞填充
            'hole_filling': {
                'enabled': True,
                'min_hole_area': 5  # 减小该值填充更小的空洞
            }
        }