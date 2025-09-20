#!/usr/bin/env python3
"""
Quick start script for image-text similarity evaluation
快速启动脚本
"""

import os
import sys
import subprocess

def check_requirements():
    """检查依赖是否安装"""
    try:
        import torch
        import clip
        import pandas
        import PIL
        print("✓ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def main():
    """主函数"""
    print("图像-文本相似度评估工具")
    print("=" * 40)
    
    # 检查依赖
    if not check_requirements():
        return
    
    # 检查输入文件
    if not os.path.exists("outputs/results.csv"):
        print("✗ 未找到输入文件: outputs/results.csv")
        return
    
    # 选择模型
    print("\n请选择评估模型:")
    print("1. CLIP (推荐用于通用图像-文本相似度)")
    print("2. BLIP (推荐用于图像-文本检索)")
    
    while True:
        choice = input("请输入选择 (1/2) 或按回车使用默认CLIP: ").strip()
        if choice == "" or choice == "1":
            model = "CLIP"
            break
        elif choice == "2":
            model = "BLIP"
            break
        else:
            print("无效选择，请重新输入")
    
    print(f"使用模型: {model}")
    
    # 运行主程序
    print("开始运行评估...")
    try:
        subprocess.run([sys.executable, "main.py", "--model", model], check=True)
        print("✓ 评估完成！")
    except subprocess.CalledProcessError as e:
        print(f"✗ 运行失败: {e}")
    except KeyboardInterrupt:
        print("\n用户中断")

if __name__ == "__main__":
    main()
