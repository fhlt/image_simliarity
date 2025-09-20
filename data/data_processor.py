"""
Data processor for handling image-text data
数据处理器
"""

import os
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, image_dir: str = "outputs/images"):
        """
        初始化数据处理器
        
        Args:
            image_dir: 图像目录
        """
        self.image_dir = image_dir
        self.stats = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'corrupted_images': 0
        }
    
    def validate_images(self, df: pd.DataFrame, image_path_column: str = 'image_path') -> pd.DataFrame:
        """
        验证图像文件
        
        Args:
            df: 包含图像路径的DataFrame
            image_path_column: 图像路径列名
            
        Returns:
            pd.DataFrame: 添加了验证结果的DataFrame
        """
        logger.info("开始验证图像文件")
        
        df_processed = df.copy()
        df_processed['image_valid'] = False
        df_processed['image_error'] = ''
        
        self.stats['total_images'] = len(df_processed)
        
        for idx, row in df_processed.iterrows():
            image_path = row[image_path_column]
            
            try:
                if not os.path.exists(image_path):
                    df_processed.at[idx, 'image_error'] = '文件不存在'
                    self.stats['invalid_images'] += 1
                    continue
                
                # 尝试打开图像
                with Image.open(image_path) as img:
                    # 检查图像格式
                    if img.format not in ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP']:
                        df_processed.at[idx, 'image_error'] = f'不支持的格式: {img.format}'
                        self.stats['invalid_images'] += 1
                        continue
                    
                    # 检查图像尺寸
                    width, height = img.size
                    if width < 32 or height < 32:
                        df_processed.at[idx, 'image_error'] = f'图像尺寸过小: {width}x{height}'
                        self.stats['invalid_images'] += 1
                        continue
                    
                    # 检查图像是否损坏
                    try:
                        img.verify()
                    except Exception as e:
                        df_processed.at[idx, 'image_error'] = f'图像损坏: {str(e)}'
                        self.stats['corrupted_images'] += 1
                        continue
                
                df_processed.at[idx, 'image_valid'] = True
                self.stats['valid_images'] += 1
                
            except Exception as e:
                df_processed.at[idx, 'image_error'] = f'验证失败: {str(e)}'
                self.stats['invalid_images'] += 1
        
        logger.info(f"图像验证完成: {self.stats}")
        return df_processed
    
    def get_valid_data(self, df: pd.DataFrame, image_path_column: str = 'image_path') -> pd.DataFrame:
        """
        获取有效的图像数据
        
        Args:
            df: 包含验证结果的DataFrame
            image_path_column: 图像路径列名
            
        Returns:
            pd.DataFrame: 有效数据
        """
        valid_df = df[df['image_valid'] == True].copy()
        logger.info(f"有效图像数: {len(valid_df)} / {len(df)}")
        return valid_df
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        获取图像信息
        
        Args:
            image_path: 图像路径
            
        Returns:
            Dict[str, Any]: 图像信息
        """
        try:
            with Image.open(image_path) as img:
                return {
                    'width': img.size[0],
                    'height': img.size[1],
                    'format': img.format,
                    'mode': img.mode,
                    'size_bytes': os.path.getsize(image_path)
                }
        except Exception as e:
            return {'error': str(e)}
    
    def add_image_info(self, df: pd.DataFrame, image_path_column: str = 'image_path') -> pd.DataFrame:
        """
        为DataFrame添加图像信息
        
        Args:
            df: DataFrame
            image_path_column: 图像路径列名
            
        Returns:
            pd.DataFrame: 添加了图像信息的DataFrame
        """
        logger.info("添加图像信息")
        
        df_with_info = df.copy()
        image_info_list = []
        
        for idx, row in df_with_info.iterrows():
            image_path = row[image_path_column]
            image_info = self.get_image_info(image_path)
            image_info_list.append(image_info)
        
        # 将图像信息添加到DataFrame
        image_info_df = pd.DataFrame(image_info_list)
        df_with_info = pd.concat([df_with_info, image_info_df], axis=1)
        
        return df_with_info
    
    def filter_by_criteria(self, 
                          df: pd.DataFrame, 
                          min_width: int = 64,
                          max_width: int = 4096,
                          min_height: int = 64,
                          max_height: int = 4096,
                          max_size_mb: float = 50.0) -> pd.DataFrame:
        """
        根据条件过滤图像
        
        Args:
            df: DataFrame
            min_width: 最小宽度
            max_width: 最大宽度
            min_height: 最小高度
            max_height: 最大高度
            max_size_mb: 最大文件大小(MB)
            
        Returns:
            pd.DataFrame: 过滤后的DataFrame
        """
        logger.info(f"根据条件过滤图像: 尺寸 {min_width}x{min_height} - {max_width}x{max_height}, 最大大小 {max_size_mb}MB")
        
        # 确保有图像信息列
        if 'width' not in df.columns:
            df = self.add_image_info(df)
        
        # 应用过滤条件
        mask = (
            (df['width'] >= min_width) &
            (df['width'] <= max_width) &
            (df['height'] >= min_height) &
            (df['height'] <= max_height) &
            (df['size_bytes'] <= max_size_mb * 1024 * 1024)
        )
        
        filtered_df = df[mask].copy()
        logger.info(f"过滤结果: {len(filtered_df)} / {len(df)} 张图像通过过滤")
        
        return filtered_df
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats.copy()
