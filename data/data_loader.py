"""
Data loader for CSV files
CSV数据加载器
"""

import os
import logging
import pandas as pd
from typing import Optional, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器"""
    
    def __init__(self, csv_path: str):
        """
        初始化数据加载器
        
        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
        self.df = None
        
    def load_csv(self) -> pd.DataFrame:
        """
        加载CSV文件
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")
            
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"成功加载CSV文件: {len(self.df)} 条记录")
            
            # 验证必要的列
            required_columns = ['url', 'caption']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"加载CSV文件失败: {e}")
            raise
    
    def validate_data(self) -> Dict[str, Any]:
        """
        验证数据质量
        
        Returns:
            Dict[str, Any]: 验证结果
        """
        if self.df is None:
            raise ValueError("请先加载数据")
        
        validation_result = {
            'total_records': len(self.df),
            'valid_urls': 0,
            'invalid_urls': 0,
            'empty_captions': 0,
            'valid_records': 0
        }
        
        for idx, row in self.df.iterrows():
            # 检查URL
            if self._is_valid_url(row['url']):
                validation_result['valid_urls'] += 1
            else:
                validation_result['invalid_urls'] += 1
            
            # 检查caption
            if pd.isna(row['caption']) or str(row['caption']).strip() == '':
                validation_result['empty_captions'] += 1
            
            # 检查完整记录
            if (self._is_valid_url(row['url']) and 
                not pd.isna(row['caption']) and 
                str(row['caption']).strip() != ''):
                validation_result['valid_records'] += 1
        
        validation_result['validity_rate'] = validation_result['valid_records'] / validation_result['total_records']
        
        logger.info(f"数据验证完成: {validation_result}")
        return validation_result
    
    def _is_valid_url(self, url: str) -> bool:
        """检查URL是否有效"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def get_valid_data(self) -> pd.DataFrame:
        """
        获取有效的数据记录
        
        Returns:
            pd.DataFrame: 有效数据
        """
        if self.df is None:
            raise ValueError("请先加载数据")
        
        # 过滤有效记录
        valid_mask = (
            self.df['url'].apply(self._is_valid_url) &
            self.df['caption'].notna() &
            (self.df['caption'].str.strip() != '')
        )
        
        valid_df = self.df[valid_mask].copy()
        logger.info(f"有效记录数: {len(valid_df)} / {len(self.df)}")
        
        return valid_df
    
    def add_image_paths(self, image_dir: str = "outputs/images") -> pd.DataFrame:
        """
        为数据添加本地图像路径
        
        Args:
            image_dir: 图像存储目录
            
        Returns:
            pd.DataFrame: 添加了图像路径的数据
        """
        if self.df is None:
            raise ValueError("请先加载数据")
        
        df_with_paths = self.df.copy()
        
        def get_image_path(url):
            """从URL生成本地图像路径"""
            filename = os.path.basename(urlparse(url).path)
            return os.path.join(image_dir, filename)
        
        df_with_paths['image_path'] = df_with_paths['url'].apply(get_image_path)
        
        return df_with_paths
