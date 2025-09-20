"""
Base similarity evaluator class
定义相似度评估器的基类接口
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class BaseSimilarityEvaluator(ABC):
    """相似度评估器基类"""
    
    def __init__(self, 
                 model_name: str,
                 device: str = None,
                 batch_size: int = 32,
                 image_size: int = 224):
        """
        初始化评估器
        
        Args:
            model_name: 模型名称
            device: 计算设备
            batch_size: 批处理大小
            image_size: 图像尺寸
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device or self._get_default_device()
        
        # 性能指标
        self.metrics = {
            'model_name': model_name,
            'device': self.device,
            'batch_size': batch_size,
            'image_size': image_size,
            'total_images': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'total_time': 0.0,
            'average_time_per_image': 0.0,
            'memory_usage_mb': 0.0,
            'gpu_utilization': 0.0
        }
        
        logger.info(f"初始化 {self.__class__.__name__} (模型: {model_name}, 设备: {self.device})")
    
    def _get_default_device(self) -> str:
        """获取默认计算设备"""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    
    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass
    
    @abstractmethod
    def compute_similarity(self, image_path: str, text: str) -> float:
        """
        计算图像-文本相似度
        
        Args:
            image_path: 图像路径
            text: 文本描述
            
        Returns:
            float: 相似度分数
        """
        pass
    
    def preprocess_image(self, image_path: str) -> Any:
        """
        预处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            预处理后的图像数据
        """
        # 子类可以重写此方法
        return image_path
    
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
            start_time = time.time()
            
            try:
                # 预处理图像
                processed_image = self.preprocess_image(item['image_path'])
                
                # 计算相似度
                similarity_score = self.compute_similarity(processed_image, item['caption'])
                
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
    
    def evaluate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        评估整个数据集
        
        Args:
            df: 包含图像路径和文本的DataFrame
            
        Returns:
            pd.DataFrame: 包含相似度分数的结果DataFrame
        """
        logger.info(f"开始处理数据集: {len(df)} 条记录")
        start_time = time.time()
        
        # 初始化结果列
        df['similarity_score'] = 0.0
        df['processing_time'] = 0.0
        df['status'] = 'pending'
        
        self.metrics['total_images'] = len(df)
        
        # 批处理
        results = []
        for i in range(0, len(df), self.batch_size):
            batch_df = df.iloc[i:i+self.batch_size].copy()
            batch_data = batch_df.to_dict('records')
            
            batch_results = self.process_batch(batch_data)
            results.extend(batch_results)
        
        # 更新DataFrame
        result_df = pd.DataFrame(results)
        
        # 计算总时间
        total_time = time.time() - start_time
        self.metrics['total_time'] = total_time
        self.metrics['average_time_per_image'] = total_time / len(df)
        
        # 记录内存使用
        self._update_memory_metrics()
        
        logger.info(f"评估完成，耗时: {total_time:.2f}秒")
        return result_df
    
    def _update_memory_metrics(self):
        """更新内存使用指标"""
        try:
            import torch
            if torch.cuda.is_available():
                self.metrics['memory_usage_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.metrics['gpu_utilization'] = torch.cuda.utilization()
        except ImportError:
            pass
    
    def save_metrics(self, metrics_path: str = "performance_metrics.txt"):
        """保存性能指标到文件"""
        try:
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write(f"{self.__class__.__name__} 性能指标\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"模型信息:\n")
                f.write(f"  模型名称: {self.metrics['model_name']}\n")
                f.write(f"  计算设备: {self.metrics['device']}\n")
                f.write(f"  批处理大小: {self.metrics['batch_size']}\n")
                f.write(f"  图像尺寸: {self.metrics['image_size']}\n\n")
                
                f.write(f"处理统计:\n")
                f.write(f"  总图像数: {self.metrics['total_images']}\n")
                f.write(f"  成功处理: {self.metrics['successful_processing']}\n")
                f.write(f"  处理失败: {self.metrics['failed_processing']}\n")
                f.write(f"  成功率: {self.metrics['successful_processing']/self.metrics['total_images']*100:.2f}%\n\n")
                
                f.write(f"性能指标:\n")
                f.write(f"  总处理时间: {self.metrics['total_time']:.2f}秒\n")
                f.write(f"  平均每张图像时间: {self.metrics['average_time_per_image']:.4f}秒\n")
                f.write(f"  内存使用: {self.metrics['memory_usage_mb']:.2f}MB\n")
                f.write(f"  GPU利用率: {self.metrics['gpu_utilization']:.2f}%\n")
                
            logger.info(f"性能指标已保存到: {metrics_path}")
            
        except Exception as e:
            logger.error(f"保存性能指标失败: {e}")
    
    def print_statistics(self):
        """打印统计信息"""
        logger.info("\n" + "="*50)
        logger.info(f"{self.__class__.__name__} 评估统计")
        logger.info("="*50)
        logger.info(f"总图像数: {self.metrics['total_images']}")
        logger.info(f"成功处理: {self.metrics['successful_processing']}")
        logger.info(f"处理失败: {self.metrics['failed_processing']}")
        logger.info(f"成功率: {self.metrics['successful_processing']/self.metrics['total_images']*100:.2f}%")
        logger.info(f"总处理时间: {self.metrics['total_time']:.2f}秒")
        logger.info(f"平均每张图像时间: {self.metrics['average_time_per_image']:.4f}秒")
        if self.metrics['memory_usage_mb'] > 0:
            logger.info(f"GPU内存使用: {self.metrics['memory_usage_mb']:.2f}MB")
            logger.info(f"GPU利用率: {self.metrics['gpu_utilization']:.2f}%")
