#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer模型集成模块，用于花朵分类任务
"""

import os
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from sklearn.metrics.pairwise import cosine_similarity

class TransformerClassifier:
    """Transformer模型分类器类"""
    
    def __init__(self, model_path=None, device=None):
        """
        初始化分类器
        
        参数:
            model_path: 模型权重路径
            device: 计算设备 (None表示自动选择)
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # 加载特征提取器
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # 加载模型
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path=None):
        """
        加载ViT模型
        
        参数:
            model_path: 模型权重路径
            
        返回:
            加载好的模型
        """
        # 加载预训练模型
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # 如果提供了权重路径，加载自定义权重
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"Successfully loaded model weights from {model_path}")
            except Exception as e:
                print(f"Error loading model weights: {str(e)}")
        
        # 移动模型到指定设备并设置为评估模式
        model = model.to(self.device).eval()
        
        return model
    
    def preprocess_image(self, image):
        """
        预处理图像
        
        参数:
            image: 输入图像 (OpenCV格式，BGR)
            
        返回:
            处理后的图像张量
        """
        # 将OpenCV图像(BGR)转换为PIL图像(RGB)
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image).convert("RGB")
        
        # 使用ViT特征提取器处理图像
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        # 移动到正确的设备
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        return inputs
    
    def extract_features(self, image):
        """
        提取图像特征
        
        参数:
            image: 输入图像
            
        返回:
            特征向量
        """
        # 预处理图像
        inputs = self.preprocess_image(image)
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 使用池化层输出作为特征向量
        features = outputs.pooler_output.cpu().numpy().flatten()
        
        return features
    
    def apply_mask(self, image_path, mask):
        """
        应用掩码到图像
        
        参数:
            image_path: 图像路径
            mask: 二值掩码
            
        返回:
            应用掩码后的图像
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        
        # 调整掩码大小以匹配图像
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 转换掩码为三通道
        if len(mask.shape) == 2 or mask.shape[2] == 1:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 应用掩码
        masked_image = cv2.bitwise_and(image, mask)
        
        return masked_image
    
    def compare_images(self, segmented_path, ground_truth_path, mask=None):
        """
        比较分割图像与真实掩码
        
        参数:
            segmented_path: 分割图像路径
            ground_truth_path: 真实掩码路径
            mask: 预计算的掩码 (可选)
            
        返回:
            余弦相似度
        """
        # 检查真实掩码是否存在
        if not os.path.exists(ground_truth_path):
            print(f"Missing ground truth: {ground_truth_path}")
            return None
        
        # 提取或使用预计算的掩码
        if mask is None:
            # 从真实掩码提取红色区域
            ground_truth = cv2.imread(ground_truth_path)
            hsv = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2HSV)
            
            # 定义HSV中的红色范围
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            
            # 创建红色掩码
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
        else:
            red_mask = mask
        
        # 应用掩码到分割图像和真实掩码
        masked_segmented = self.apply_mask(segmented_path, red_mask)
        masked_ground_truth = self.apply_mask(ground_truth_path, red_mask)
        
        # 提取特征
        seg_features = self.extract_features(masked_segmented)
        gt_features = self.extract_features(masked_ground_truth)
        
        # 计算余弦相似度
        similarity = cosine_similarity([seg_features], [gt_features])[0][0]
        
        return similarity
    
    def evaluate_dataset(self, segmented_folder, ground_truth_folder, threshold=0.9):
        """
        评估整个数据集
        
        参数:
            segmented_folder: 分割结果文件夹
            ground_truth_folder: 真实掩码文件夹
            threshold: 准确率阈值
            
        返回:
            评估结果DataFrame和整体指标
        """
        results = []
        
        # 遍历分割结果文件夹中的所有图像
        for filename in tqdm(os.listdir(segmented_folder)):
            if filename.lower().endswith((".jpg", ".png")):
                segmented_path = os.path.join(segmented_folder, filename)
                ground_truth_path = os.path.join(ground_truth_folder, os.path.splitext(filename)[0] + ".png")
                
                # 如果找不到对应的真实掩码，尝试其他扩展名
                if not os.path.exists(ground_truth_path):
                    ground_truth_path = os.path.join(ground_truth_folder, os.path.splitext(filename)[0] + ".jpg")
                
                # 计算相似度
                similarity = self.compare_images(segmented_path, ground_truth_path)
                
                if similarity is not None:
                    # 根据阈值判断准确率
                    accuracy = 1 if similarity >= threshold else 0
                    results.append([filename, similarity, accuracy])
        
        # 创建DataFrame
        metrics_df = pd.DataFrame(results, columns=["Image", "Cosine Similarity", "Accuracy"])
        
        # 计算整体准确率和相似度
        overall_similarity = metrics_df['Cosine Similarity'].mean() * 100
        overall_accuracy = metrics_df['Accuracy'].mean() * 100
        
        return metrics_df, overall_similarity, overall_accuracy


def main():
    """主函数示例"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Flower Classification with Transformer Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained transformer model')
    parser.add_argument('--segmented_folder', type=str, required=True,
                        help='Directory containing segmented images')
    parser.add_argument('--ground_truth_folder', type=str, required=True,
                        help='Directory containing ground truth masks')
    parser.add_argument('--output_csv', type=str, default='classification_results.csv',
                        help='CSV file to save results')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Similarity threshold for accuracy')
    args = parser.parse_args()
    
    # 创建分类器
    classifier = TransformerClassifier(model_path=args.model_path)
    
    # 评估数据集
    metrics_df, overall_similarity, overall_accuracy = classifier.evaluate_dataset(
        args.segmented_folder, 
        args.ground_truth_folder, 
        threshold=args.threshold
    )
    
    # 保存结果
    metrics_df.to_csv(args.output_csv, index=False)
    
    # 显示结果
    print(f"\nOverall Segmentation Accuracy: {overall_accuracy:.2f}%")
    print(f"Overall Cosine Similarity: {overall_similarity:.2f}%")


if __name__ == "__main__":
    main()