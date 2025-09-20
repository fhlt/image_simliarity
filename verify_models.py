#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify model configurations
验证模型配置
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_clip_config, get_blip_config, get_model_config

def main():
    """验证模型配置"""
    print("验证模型配置")
    print("=" * 40)
    
    # 验证CLIP配置
    clip_config = get_clip_config()
    print(f"CLIP配置:")
    print(f"  模型名称: {clip_config.model.name}")
    print(f"  模型路径: {clip_config.model.model_name}")
    print(f"  图像尺寸: {clip_config.model.image_size}")
    
    # 验证BLIP配置
    blip_config = get_blip_config()
    print(f"\nBLIP配置:")
    print(f"  模型名称: {blip_config.model.name}")
    print(f"  模型路径: {blip_config.model.model_name}")
    print(f"  图像尺寸: {blip_config.model.image_size}")
    
    # 验证模型选择
    print(f"\n模型选择测试:")
    config1 = get_model_config("CLIP")
    config2 = get_model_config("BLIP")
    print(f"  CLIP选择: {config1.model.model_name}")
    print(f"  BLIP选择: {config2.model.model_name}")
    
    # 验证模型路径格式
    print(f"\n模型路径验证:")
    clip_path = clip_config.model.model_name
    blip_path = blip_config.model.model_name
    
    if clip_path.startswith("openai/"):
        print(f"  ✓ CLIP路径格式正确: {clip_path}")
    else:
        print(f"  ✗ CLIP路径格式错误: {clip_path}")
    
    if blip_path.startswith("Salesforce/"):
        print(f"  ✓ BLIP路径格式正确: {blip_path}")
    else:
        print(f"  ✗ BLIP路径格式错误: {blip_path}")
    
    print(f"\n验证完成！")

if __name__ == "__main__":
    main()
