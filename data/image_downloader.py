"""
Image downloader for downloading images from URLs
图像下载器
"""

import os
import logging
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class ImageDownloader:
    """图像下载器"""
    
    def __init__(self, 
                 download_dir: str = "outputs/images",
                 timeout: int = 30,
                 max_retries: int = 3,
                 delay: float = 0.1):
        """
        初始化图像下载器
        
        Args:
            download_dir: 下载目录
            timeout: 请求超时时间
            max_retries: 最大重试次数
            delay: 请求间延迟
        """
        self.download_dir = download_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.delay = delay
        
        # 创建下载目录
        os.makedirs(download_dir, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_urls': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'skipped_existing': 0,
            'total_time': 0.0
        }
    
    def download_image(self, url: str, filename: str = None) -> Dict[str, Any]:
        """
        下载单个图像
        
        Args:
            url: 图像URL
            filename: 文件名（可选）
            
        Returns:
            Dict[str, Any]: 下载结果
        """
        if filename is None:
            filename = os.path.basename(urlparse(url).path)
        
        filepath = os.path.join(self.download_dir, filename)
        
        # 检查文件是否已存在
        if os.path.exists(filepath):
            logger.debug(f"图像已存在，跳过: {filename}")
            self.stats['skipped_existing'] += 1
            return {
                'success': True,
                'filepath': filepath,
                'filename': filename,
                'skipped': True
            }
        
        # 下载图像
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # 检查内容类型
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    logger.warning(f"URL不是图像文件: {url} (类型: {content_type})")
                    if attempt == self.max_retries - 1:
                        self.stats['failed_downloads'] += 1
                        return {
                            'success': False,
                            'error': f'不是图像文件 (类型: {content_type})',
                            'filename': filename
                        }
                    continue
                
                # 保存文件
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # 验证文件大小
                if os.path.getsize(filepath) == 0:
                    os.remove(filepath)
                    raise ValueError("下载的文件为空")
                
                logger.debug(f"成功下载: {filename}")
                self.stats['successful_downloads'] += 1
                
                return {
                    'success': True,
                    'filepath': filepath,
                    'filename': filename,
                    'size': os.path.getsize(filepath)
                }
                
            except Exception as e:
                logger.warning(f"下载失败 (尝试 {attempt + 1}/{self.max_retries}): {url} - {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (2 ** attempt))  # 指数退避
                else:
                    self.stats['failed_downloads'] += 1
                    return {
                        'success': False,
                        'error': str(e),
                        'filename': filename
                    }
    
    def download_batch(self, urls: List[str], progress_bar: bool = True) -> List[Dict[str, Any]]:
        """
        批量下载图像
        
        Args:
            urls: URL列表
            progress_bar: 是否显示进度条
            
        Returns:
            List[Dict[str, Any]]: 下载结果列表
        """
        logger.info(f"开始批量下载 {len(urls)} 张图像")
        start_time = time.time()
        
        self.stats['total_urls'] = len(urls)
        results = []
        
        # 使用进度条
        if progress_bar:
            urls_iter = tqdm(urls, desc="下载图像")
        else:
            urls_iter = urls
        
        for url in urls_iter:
            filename = os.path.basename(urlparse(url).path)
            result = self.download_image(url, filename)
            results.append(result)
            
            # 添加延迟避免请求过于频繁
            if self.delay > 0:
                time.sleep(self.delay)
        
        # 更新统计信息
        self.stats['total_time'] = time.time() - start_time
        
        logger.info(f"批量下载完成: {self.stats}")
        return results
    
    def download_from_dataframe(self, df, url_column: str = 'url', progress_bar: bool = True) -> List[Dict[str, Any]]:
        """
        从DataFrame下载图像
        
        Args:
            df: 包含URL的DataFrame
            url_column: URL列名
            progress_bar: 是否显示进度条
            
        Returns:
            List[Dict[str, Any]]: 下载结果列表
        """
        urls = df[url_column].tolist()
        return self.download_batch(urls, progress_bar)
    
    def get_download_stats(self) -> Dict[str, Any]:
        """获取下载统计信息"""
        return self.stats.copy()
    
    def cleanup_failed_downloads(self):
        """清理失败的下载文件"""
        if not os.path.exists(self.download_dir):
            return
        
        cleaned = 0
        for filename in os.listdir(self.download_dir):
            filepath = os.path.join(self.download_dir, filename)
            if os.path.isfile(filepath) and os.path.getsize(filepath) == 0:
                os.remove(filepath)
                cleaned += 1
        
        if cleaned > 0:
            logger.info(f"清理了 {cleaned} 个空文件")
