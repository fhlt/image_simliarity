#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify project structure
项目结构验证脚本
"""

import os
import sys
import importlib

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        # 测试配置模块
        import config
        print("✓ config 模块导入成功")
        
        # 测试数据模块
        from data import DataLoader, ImageDownloader, DataProcessor
        print("✓ data 模块导入成功")
        
        # 测试模型模块
        from models import CLIPSimilarityEvaluator, BaseSimilarityEvaluator
        print("✓ models 模块导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n测试文件结构...")
    
    required_files = [
        "main.py",
        "config.py",
        "requirements.txt",
        "README.md",
        "models/__init__.py",
        "models/base_evaluator.py",
        "models/clip_evaluator.py",
        "data/__init__.py",
        "data/data_loader.py",
        "data/image_downloader.py",
        "data/data_processor.py",
        "outputs/results.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"\n✗ 缺少文件: {missing_files}")
        return False
    
    return True

def test_config_system():
    """测试配置系统"""
    print("\n测试配置系统...")
    
    try:
        from config import get_default_config, create_config_file
        
        # 测试默认配置
        config = get_default_config()
        print("✓ 默认配置创建成功")
        
        # 测试配置保存
        create_config_file("test_config.json")
        print("✓ 配置文件保存成功")
        
        # 清理测试文件
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")
        
        return True
    except Exception as e:
        print(f"✗ 配置系统测试失败: {e}")
        return False

def test_data_loader():
    """测试数据加载器"""
    print("\n测试数据加载器...")
    
    try:
        from data import DataLoader
        
        if not os.path.exists("outputs/results.csv"):
            print("✗ 输入文件不存在，跳过数据加载器测试")
            return True
        
        loader = DataLoader("outputs/results.csv")
        df = loader.load_csv()
        print(f"✓ 数据加载成功: {len(df)} 条记录")
        
        validation_result = loader.validate_data()
        print(f"✓ 数据验证完成: {validation_result['validity_rate']:.2%} 有效")
        
        return True
    except Exception as e:
        print(f"✗ 数据加载器测试失败: {e}")
        return False

def main():
    """主函数"""
    print("项目结构验证")
    print("=" * 50)
    
    tests = [
        ("文件结构", test_file_structure),
        ("模块导入", test_imports),
        ("配置系统", test_config_system),
        ("数据加载器", test_data_loader)
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
        print("🎉 所有测试通过！项目结构正确。")
        print("\n可以开始使用:")
        print("  python main.py                    # 运行评估")
        print("  python main.py --create-config    # 创建配置文件")
        print("  python run.py                     # 快速启动")
    else:
        print("❌ 部分测试失败，请检查项目结构。")

if __name__ == "__main__":
    main()
