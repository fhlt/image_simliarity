#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test BLIP fix
测试BLIP修复
"""

import sys
import os
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import BLIPSimilarityEvaluator

def test_blip_fix():
    """测试BLIP修复"""
    print("测试BLIP修复")
    print("=" * 40)
    
    try:
        # 创建BLIP评估器
        print("创建BLIP评估器...")
        evaluator = BLIPSimilarityEvaluator(
            model_name="Salesforce/blip-itm-base-coco",
            device="cpu",  # 使用CPU避免GPU问题
            batch_size=1,
            image_size=384
        )
        print("✓ BLIP评估器创建成功")
        
        # 创建测试数据
        print("\n创建测试数据...")
        test_data = [
            {
                'image_path': 'outputs/images/test1.jpg',  # 假设的图像路径
                'caption': 'a test image'
            }
        ]
        
        # 测试相似度计算（不实际运行，只测试代码结构）
        print("测试相似度计算代码结构...")
        
        # 模拟图像张量
        import torch
        dummy_image_tensor = torch.randn(1, 3, 384, 384)
        
        # 测试相似度计算
        similarity = evaluator.compute_similarity(dummy_image_tensor, "a test image")
        print(f"相似度分数: {similarity}")
        
        if similarity > 0:
            print("✓ BLIP相似度计算成功")
        else:
            print("⚠ BLIP相似度计算返回0，可能是模型未正确加载")
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_blip_fix()
