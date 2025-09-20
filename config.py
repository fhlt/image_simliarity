"""
Configuration system for image-text similarity evaluation
配置系统
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "CLIP"  # 支持 "CLIP", "BLIP"
    model_name: str = "openai/clip-vit-base-patch32"  # CLIP: "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14" | BLIP: "Salesforce/blip-itm-base-coco", "Salesforce/blip-itm-large-coco"
    device: Optional[str] = None
    batch_size: int = 32
    image_size: int = 224  # CLIP默认224，BLIP默认384


@dataclass
class DataConfig:
    """数据配置"""
    input_csv: str = "outputs/results.csv"
    output_csv: str = "outputs/results_with_scores.csv"
    image_dir: str = "outputs/images"
    download_images: bool = True
    validate_images: bool = True
    min_image_width: int = 64
    max_image_width: int = 4096
    min_image_height: int = 64
    max_image_height: int = 4096
    max_image_size_mb: float = 50.0


@dataclass
class DownloadConfig:
    """下载配置"""
    timeout: int = 30
    max_retries: int = 3
    delay: float = 0.1
    progress_bar: bool = True


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    log_file: str = "evaluation.log"
    console_output: bool = True


@dataclass
class EvaluationConfig:
    """评估配置"""
    model: ModelConfig
    data: DataConfig
    download: DownloadConfig
    logging: LoggingConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvaluationConfig':
        """从字典创建配置"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            download=DownloadConfig(**config_dict.get('download', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def save(self, config_path: str):
        """保存配置到文件"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到: {config_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    @classmethod
    def load(cls, config_path: str) -> 'EvaluationConfig':
        """从文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            logger.info(f"配置已从 {config_path} 加载")
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise


def get_default_config() -> EvaluationConfig:
    """获取默认配置"""
    return EvaluationConfig(
        model=ModelConfig(),
        data=DataConfig(),
        download=DownloadConfig(),
        logging=LoggingConfig()
    )


def get_clip_config() -> EvaluationConfig:
    """获取CLIP模型配置"""
    return EvaluationConfig(
        model=ModelConfig(
            name="CLIP",
            model_name="openai/clip-vit-base-patch32",
            image_size=224
        ),
        data=DataConfig(),
        download=DownloadConfig(),
        logging=LoggingConfig()
    )


def get_blip_config() -> EvaluationConfig:
    """获取BLIP模型配置"""
    return EvaluationConfig(
        model=ModelConfig(
            name="BLIP",
            model_name="Salesforce/blip-itm-base-coco",
            image_size=384
        ),
        data=DataConfig(),
        download=DownloadConfig(),
        logging=LoggingConfig()
    )


def get_model_config(model_name: str) -> EvaluationConfig:
    """根据模型名称获取配置"""
    if model_name.upper() == "CLIP":
        return get_clip_config()
    elif model_name.upper() == "BLIP":
        return get_blip_config()
    else:
        logger.warning(f"未知的模型名称: {model_name}，使用默认配置")
        return get_default_config()


def create_config_file(config_path: str = "config.json"):
    """创建默认配置文件"""
    config = get_default_config()
    config.save(config_path)
    logger.info(f"默认配置文件已创建: {config_path}")


def load_config(config_path: str = "config.json") -> EvaluationConfig:
    """加载配置文件，如果不存在则创建默认配置"""
    if not os.path.exists(config_path):
        logger.info(f"配置文件不存在，创建默认配置: {config_path}")
        create_config_file(config_path)
    
    return EvaluationConfig.load(config_path)
