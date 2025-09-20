"""
BLIP-based similarity evaluator
基于BLIP模型的图像-文本相似度评估器
"""

import os
import logging
from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForImageTextRetrieval

from .base_evaluator import BaseSimilarityEvaluator

logger = logging.getLogger(__name__)


class BLIPSimilarityEvaluator(BaseSimilarityEvaluator):
    """BLIP-based图像-文本相似度评估器"""
    
    def __init__(self, 
                 model_name: str = "Salesforce/blip-itm-base-coco",
                 device: str = None,
                 batch_size: int = 32,
                 image_size: int = 384):
        """
        初始化BLIP评估器
        
        Args:
            model_name: BLIP模型名称
            device: 计算设备
            batch_size: 批处理大小
            image_size: 图像尺寸
        """
        super().__init__(model_name, device, batch_size, image_size)
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """加载BLIP模型"""
        try:
            logger.info(f"正在加载BLIP模型: {self.model_name}")
            import time
            start_time = time.time()
            
            # 加载BLIP处理器和模型
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForImageTextRetrieval.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"BLIP模型加载完成，耗时: {load_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"BLIP模型加载失败: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        try:
            image = Image.open(image_path).convert('RGB')
            # 使用BLIP处理器预处理图像
            inputs = self.processor(images=image, return_tensors="pt")
            image_tensor = inputs.pixel_values.to(self.device)
            return image_tensor
        except Exception as e:
            logger.warning(f"图像预处理失败 {image_path}: {e}")
            return None
    
    def compute_similarity(self, image_tensor: torch.Tensor, text: str) -> float:
        """
        计算图像-文本相似度
        
        Args:
            image_tensor: 预处理后的图像张量
            text: 文本描述
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            if image_tensor is None:
                return 0.0
            
            # 使用BLIP处理器处理文本
            text_inputs = self.processor(text=text, return_tensors="pt")
            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)
            
            with torch.no_grad():
                # 获取图像和文本特征
                outputs = self.model(
                    pixel_values=image_tensor,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 计算相似度分数
                # BLIP ITM模型通常输出logits，形状为[batch_size, 2]
                # 其中logits[:, 1]表示匹配分数，logits[:, 0]表示不匹配分数
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    if logits.dim() == 2 and logits.size(1) == 2:
                        # 对于二分类输出，取匹配分数（索引1）
                        match_logits = logits[:, 1]
                        similarity = torch.sigmoid(match_logits).squeeze()
                    else:
                        # 对于其他形状的logits，直接使用
                        similarity = torch.sigmoid(logits).squeeze()
                elif hasattr(outputs, 'logits_per_image'):
                    # 如果有logits_per_image属性
                    logits = outputs.logits_per_image
                    similarity = torch.sigmoid(logits).squeeze()
                elif hasattr(outputs, 'itm_score'):
                    # 如果有itm_score属性
                    similarity = torch.sigmoid(outputs.itm_score).squeeze()
                else:
                    # 如果都没有，记录警告并返回默认值
                    logger.warning(f"无法从BLIP输出中提取相似度分数，输出属性: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                    return 0.5  # 返回中性分数
                
                return similarity.item()
                
        except Exception as e:
            logger.warning(f"相似度计算失败: {e}")
            return 0.0
    
    def process_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """
        批处理图像-文本对
        
        Args:
            batch_data: 批处理数据列表
            
        Returns:
            List[Dict]: 处理结果列表
        """
        results = []
        
        for item in batch_data:
            import time
            start_time = time.time()
            
            try:
                # 预处理图像
                image_tensor = self.preprocess_image(item['image_path'])
                if image_tensor is None:
                    item['similarity_score'] = 0.0
                    item['processing_time'] = 0.0
                    item['status'] = 'preprocessing_failed'
                    results.append(item)
                    continue
                
                # 计算相似度
                similarity_score = self.compute_similarity(image_tensor, item['caption'])
                
                processing_time = time.time() - start_time
                
                item['similarity_score'] = similarity_score
                item['processing_time'] = processing_time
                item['status'] = 'success'
                
                self.metrics['successful_processing'] += 1
                
            except Exception as e:
                logger.warning(f"处理失败 {item.get('image_path', 'unknown')}: {e}")
                item['similarity_score'] = 0.0
                item['processing_time'] = 0.0
                item['status'] = 'error'
                self.metrics['failed_processing'] += 1
            
            results.append(item)
        
        return results
