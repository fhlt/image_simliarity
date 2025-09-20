#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of the image-text similarity evaluation tool
使用示例
"""

import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_default_config, get_clip_config, get_blip_config
from data import DataLoader, ImageDownloader, DataProcessor
from models import CLIPSimilarityEvaluator, BLIPSimilarityEvaluator

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 1. 加载数据
    print("1. 加载数据...")
    loader = DataLoader("outputs/results.csv")
    df = loader.load_csv()
    print(f"   加载了 {len(df)} 条记录")
    
    # 2. 验证数据
    print("2. 验证数据...")
    validation_result = loader.validate_data()
    print(f"   有效记录: {validation_result['valid_records']} / {validation_result['total_records']}")
    
    # 3. 获取有效数据
    df_valid = loader.get_valid_data()
    print(f"   有效数据: {len(df_valid)} 条")
    
    # 4. 添加图像路径
    print("3. 添加图像路径...")
    df_with_paths = loader.add_image_paths("outputs/images")
    print(f"   添加了图像路径列")
    
    # 5. 创建CLIP评估器
    print("4. 创建CLIP评估器...")
    evaluator = CLIPSimilarityEvaluator(
        model_name="openai/clip-vit-base-patch32",
        device="cpu",  # 使用CPU模式
        batch_size=4   # 小批量
    )
    print("   CLIP评估器创建完成")
    
    # 6. 运行评估（仅处理前5条记录作为示例）
    print("5. 运行评估...")
    df_sample = df_with_paths.head(5).copy()
    result_df = evaluator.evaluate_dataset(df_sample)
    print(f"   评估完成: {len(result_df)} 条记录")
    
    # 7. 显示结果
    print("6. 结果统计:")
    if 'similarity_score' in result_df.columns:
        scores = result_df['similarity_score']
        print(f"   平均分数: {scores.mean():.4f}")
        print(f"   最高分数: {scores.max():.4f}")
        print(f"   最低分数: {scores.min():.4f}")
    
    return result_df

def example_with_download():
    """包含图像下载的示例"""
    print("\n=== 包含图像下载的示例 ===")
    
    # 1. 加载数据
    loader = DataLoader("outputs/results.csv")
    df = loader.load_csv()
    df_valid = loader.get_valid_data()
    
    # 2. 下载图像
    print("1. 下载图像...")
    downloader = ImageDownloader(
        download_dir="outputs/images",
        timeout=10,
        max_retries=2,
        delay=0.1
    )
    
    # 只下载前3张图像作为示例
    df_sample = df_valid.head(3)
    download_results = downloader.download_from_dataframe(df_sample)
    
    successful = sum(1 for r in download_results if r['success'])
    print(f"   下载完成: {successful}/{len(download_results)} 成功")
    
    # 3. 验证图像
    print("2. 验证图像...")
    processor = DataProcessor("outputs/images")
    df_with_paths = loader.add_image_paths("outputs/images")
    df_validated = processor.validate_images(df_with_paths)
    df_valid_images = processor.get_valid_data(df_validated)
    
    print(f"   有效图像: {len(df_valid_images)} 张")
    
    return df_valid_images

def example_config_usage():
    """配置使用示例"""
    print("\n=== 配置使用示例 ===")
    
    # 1. 获取不同模型配置
    clip_config = get_clip_config()
    blip_config = get_blip_config()
    
    print("1. CLIP配置:")
    print(f"   模型: {clip_config.model.name} ({clip_config.model.model_name})")
    print(f"   图像尺寸: {clip_config.model.image_size}")
    
    print("2. BLIP配置:")
    print(f"   模型: {blip_config.model.name} ({blip_config.model.model_name})")
    print(f"   图像尺寸: {blip_config.model.image_size}")
    
    # 3. 修改配置
    config = get_default_config()
    config.model.batch_size = 8
    config.model.device = "cpu"
    config.data.download_images = False
    
    print("3. 修改后的配置:")
    print(f"   批处理大小: {config.model.batch_size}")
    print(f"   设备: {config.model.device}")
    print(f"   下载图像: {config.data.download_images}")
    
    # 4. 保存配置
    config.save("example_config.json")
    print("4. 配置已保存到 example_config.json")
    
    return config


def example_model_comparison():
    """模型比较示例"""
    print("\n=== 模型比较示例 ===")
    
    print("CLIP vs BLIP 模型特点:")
    print("CLIP:")
    print("  - 通用图像-文本相似度计算")
    print("  - 支持多种预训练模型")
    print("  - 计算速度快")
    print("  - 适合大规模数据处理")
    
    print("BLIP:")
    print("  - 专门用于图像-文本检索")
    print("  - 更好的语义理解能力")
    print("  - 计算相对较慢")
    print("  - 适合高质量评估")
    
    return True

def main():
    """主函数"""
    print("图像-文本相似度评估工具使用示例")
    print("=" * 50)
    
    try:
        # 检查输入文件
        if not os.path.exists("outputs/results.csv"):
            print("错误: 未找到输入文件 outputs/results.csv")
            return
        
        # 运行示例
        example_config_usage()
        example_model_comparison()
        example_basic_usage()
        # example_with_download()  # 取消注释以运行下载示例
        
        print("\n示例运行完成！")
        print("\n要运行完整评估，请使用:")
        print("  python main.py --model CLIP    # 使用CLIP模型")
        print("  python main.py --model BLIP    # 使用BLIP模型")
        print("  python run.py                  # 交互式选择模型")
        print("  python main.py --list-models   # 查看支持的模型")
        
    except Exception as e:
        print(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
