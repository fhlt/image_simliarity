"""
Data module for image-text similarity evaluation
包含数据加载、下载和预处理功能
"""

from .data_loader import DataLoader
from .image_downloader import ImageDownloader
from .data_processor import DataProcessor

__all__ = ['DataLoader', 'ImageDownloader', 'DataProcessor']
