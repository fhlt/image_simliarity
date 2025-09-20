#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for model integration
模型集成测试脚本
"""

import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_clip_config, get_blip_config, get_model_config
from models import CLIPSimilarityEvaluator, BLIPSimilarityEvaluator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config_system():
    """测试配置系统"""
    print("=== 测试配置系统 ===")
    
    try:
        # 测试CLIP配置
        clip_config = get_clip_config()
        print(f"✓ CLIP配置: {clip_config.model.name} - {clip_config.model.model_name}")
        
        # 测试BLIP配置
        blip_config = get_blip_config()
        print(f"✓ BLIP配置: {blip_config.model.name} - {blip_config.model.model_name}")
        
        # 测试模型选择
        config1 = get_model_config("CLIP")
        config2 = get_model_config("BLIP")
        config3 = get_model_config("UNKNOWN")
        
        print(f"✓ 模型选择测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 配置系统测试失败: {e}")
        return False


def test_model_imports():
    """测试模型导入"""
    print("\n=== 测试模型导入 ===")
    
    try:
        from models import CLIPSimilarityEvaluator, BLIPSimilarityEvaluator, BaseSimilarityEvaluator
        print("✓ 模型导入成功")
        
        # 检查类是否存在
        assert hasattr(CLIPSimilarityEvaluator, 'compute_similarity')
        assert hasattr(BLIPSimilarityEvaluator, 'compute_similarity')
        assert hasattr(BaseSimilarityEvaluator, 'evaluate_dataset')
        
        print("✓ 模型类方法检查通过")
        return True
        
    except Exception as e:
        print(f"✗ 模型导入测试失败: {e}")
        return False


def test_model_creation():
    """测试模型创建（不加载实际模型）"""
    print("\n=== 测试模型创建 ===")
    
    try:
        # 测试CLIP评估器创建（使用CPU模式避免GPU依赖）
        clip_evaluator = CLIPSimilarityEvaluator(
            model_name="openai/clip-vit-base-patch32",
            device="cpu",
            batch_size=4,
            image_size=224
        )
        print("✓ CLIP评估器创建成功")
        
        # 测试BLIP评估器创建（使用CPU模式避免GPU依赖）
        blip_evaluator = BLIPSimilarityEvaluator(
            model_name="Salesforce/blip-itm-base-coco",
            device="cpu",
            batch_size=4,
            image_size=384
        )
        print("✓ BLIP评估器创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建测试失败: {e}")
        return False


def test_main_arguments():
    """测试主程序参数解析"""
    print("\n=== 测试主程序参数 ===")
    
    try:
        import subprocess
        
        # 测试帮助信息
        result = subprocess.run([sys.executable, "main.py", "--help"], 
                              capture_output=True, text=True)
        if "CLIP" in result.stdout and "BLIP" in result.stdout:
            print("✓ 帮助信息包含模型选择")
        else:
            print("✗ 帮助信息缺少模型选择")
            return False
        
        # 测试列出模型
        result = subprocess.run([sys.executable, "main.py", "--list-models"], 
                              capture_output=True, text=True)
        if "CLIP" in result.stdout and "BLIP" in result.stdout:
            print("✓ 模型列表显示正常")
        else:
            print("✗ 模型列表显示异常")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 主程序参数测试失败: {e}")
        return False


def test_model_comparison():
    """测试模型比较功能"""
    print("\n=== 测试模型比较 ===")
    
    try:
        # 创建两个配置进行比较
        clip_config = get_clip_config()
        blip_config = get_blip_config()
        
        print(f"CLIP配置:")
        print(f"  模型名称: {clip_config.model.name}")
        print(f"  模型文件: {clip_config.model.model_name}")
        print(f"  图像尺寸: {clip_config.model.image_size}")
        
        print(f"BLIP配置:")
        print(f"  模型名称: {blip_config.model.name}")
        print(f"  模型文件: {blip_config.model.model_name}")
        print(f"  图像尺寸: {blip_config.model.image_size}")
        
        # 验证配置差异
        assert clip_config.model.name != blip_config.model.name
        assert clip_config.model.image_size != blip_config.model.image_size
        
        print("✓ 模型配置比较通过")
        return True
        
    except Exception as e:
        print(f"✗ 模型比较测试失败: {e}")
        return False


def main():
    """主函数"""
    print("模型集成测试")
    print("=" * 50)
    
    tests = [
        ("配置系统", test_config_system),
        ("模型导入", test_model_imports),
        ("模型创建", test_model_creation),
        ("主程序参数", test_main_arguments),
        ("模型比较", test_model_comparison)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}测试:")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✓ {test_name}测试通过")
        else:
            print(f"✗ {test_name}测试失败")
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！BLIP集成成功。")
        print("\n使用方法:")
        print("  python main.py --model CLIP    # 使用CLIP模型")
        print("  python main.py --model BLIP    # 使用BLIP模型")
        print("  python main.py --list-models   # 列出支持的模型")
    else:
        print("❌ 部分测试失败，请检查集成。")

if __name__ == "__main__":
    main()
