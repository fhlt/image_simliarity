"""
CLIP-based similarity evaluator
基于CLIP模型的图像-文本相似度评估器
"""

import os
import logging
from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from .base_evaluator import BaseSimilarityEvaluator

logger = logging.getLogger(__name__)


class CLIPSimilarityEvaluator(BaseSimilarityEvaluator):
    """CLIP-based图像-文本相似度评估器"""
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = None,
                 batch_size: int = 32,
                 image_size: int = 224):
        """
        初始化CLIP评估器
        
        Args:
            model_name: CLIP模型名称
            device: 计算设备
            batch_size: 批处理大小
            image_size: 图像尺寸
        """
        super().__init__(model_name, device, batch_size, image_size)
        self.model = None
        self.preprocess = None
        self.max_text_length = 77
        self.load_model()
    
    def load_model(self):
        """加载CLIP模型"""
        try:
            logger.info(f"正在加载CLIP模型: {self.model_name}")
            import time
            start_time = time.time()
            
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.preprocess = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"CLIP模型加载完成，耗时: {load_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"CLIP模型加载失败: {e}")
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
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
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
            
            # 编码文本
            inputs = self.preprocess(
                text=[text],
                images=image_tensor,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_text_length
            ).to(self.device)
            
            with torch.no_grad():
                # 获取图像和文本特征
                outputs = self.model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                # 归一化特征
                image_features = F.normalize(image_features, p=2, dim=-1)
                text_features = F.normalize(text_features, p=2, dim=-1)
                
                # 计算余弦相似度
                similarity = torch.sum(image_features * text_features, dim=-1)
                
                return float(similarity.item())
                
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
                image = Image.open(item['image_path']).convert('RGB')
                
                # 计算相似度
                similarity_score = self.compute_similarity(image, item['caption'])
                
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
