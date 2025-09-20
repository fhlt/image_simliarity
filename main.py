#!/usr/bin/env python3
"""
Main entry point for image-text similarity evaluation
图像-文本相似度评估主程序
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import load_config, create_config_file, get_model_config
from data import DataLoader, ImageDownloader, DataProcessor
from models import CLIPSimilarityEvaluator, BLIPSimilarityEvaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SimilarityEvaluationPipeline:
    """相似度评估管道"""
    
    def __init__(self, config_path: str = "config.json", model_name: str = None):
        """
        初始化评估管道
        
        Args:
            config_path: 配置文件路径
            model_name: 模型名称 (CLIP/BLIP)，如果指定则覆盖配置文件设置
        """
        if model_name:
            self.config = get_model_config(model_name)
            logger.info(f"使用指定模型: {model_name}")
        else:
            self.config = load_config(config_path)
        self.setup_logging()
        
        # 初始化组件
        self.data_loader = DataLoader(self.config.data.input_csv)
        self.image_downloader = ImageDownloader(
            download_dir=self.config.data.image_dir,
            timeout=self.config.download.timeout,
            max_retries=self.config.download.max_retries,
            delay=self.config.download.delay
        )
        self.data_processor = DataProcessor(self.config.data.image_dir)
        
        # 初始化模型
        self.model = self._create_model()
        
        logger.info("评估管道初始化完成")
    
    def setup_logging(self):
        """设置日志"""
        # 创建日志目录
        log_dir = os.path.dirname(self.config.logging.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志处理器
        handlers = []
        
        if self.config.logging.console_output:
            handlers.append(logging.StreamHandler(sys.stdout))
        
        if self.config.logging.log_file:
            handlers.append(logging.FileHandler(self.config.logging.log_file))
        
        # 重新配置日志
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True
        )
    
    def _create_model(self):
        """创建模型实例"""
        if self.config.model.name.upper() == "CLIP":
            return CLIPSimilarityEvaluator(
                model_name=self.config.model.model_name,
                device=self.config.model.device,
                batch_size=self.config.model.batch_size,
                image_size=self.config.model.image_size
            )
        elif self.config.model.name.upper() == "BLIP":
            return BLIPSimilarityEvaluator(
                model_name=self.config.model.model_name,
                device=self.config.model.device,
                batch_size=self.config.model.batch_size,
                image_size=self.config.model.image_size
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.config.model.name}。支持的模型: CLIP, BLIP")
    
    def run(self):
        """运行评估管道"""
        try:
            logger.info("开始运行相似度评估管道")
            
            # 1. 加载数据
            logger.info("步骤 1: 加载数据")
            df = self.data_loader.load_csv()
            
            # 验证数据
            validation_result = self.data_loader.validate_data()
            logger.info(f"数据验证结果: {validation_result}")
            
            # 获取有效数据
            df_valid = self.data_loader.get_valid_data()
            if len(df_valid) == 0:
                logger.error("没有有效的数据记录")
                return
            
            # 2. 下载图像（如果需要）
            if self.config.data.download_images:
                logger.info("步骤 2: 下载图像")
                download_results = self.image_downloader.download_from_dataframe(
                    df_valid, 
                    progress_bar=self.config.download.progress_bar
                )
                
                # 检查下载结果
                successful_downloads = sum(1 for r in download_results if r['success'])
                logger.info(f"图像下载完成: {successful_downloads}/{len(download_results)} 成功")
            
            # 3. 添加图像路径
            logger.info("步骤 3: 添加图像路径")
            df_with_paths = self.data_loader.add_image_paths(self.config.data.image_dir)
            
            # 4. 验证图像（如果需要）
            if self.config.data.validate_images:
                logger.info("步骤 4: 验证图像")
                df_validated = self.data_processor.validate_images(df_with_paths)
                
                # 过滤有效图像
                df_valid_images = self.data_processor.get_valid_data(df_validated)
                
                # 根据条件过滤
                df_filtered = self.data_processor.filter_by_criteria(
                    df_valid_images,
                    min_width=self.config.data.min_image_width,
                    max_width=self.config.data.max_image_width,
                    min_height=self.config.data.min_image_height,
                    max_height=self.config.data.max_image_height,
                    max_size_mb=self.config.data.max_image_size_mb
                )
            else:
                df_filtered = df_with_paths
            
            if len(df_filtered) == 0:
                logger.error("没有有效的图像数据")
                return
            
            logger.info(f"准备处理 {len(df_filtered)} 张图像")
            
            # 5. 运行相似度评估
            logger.info("步骤 5: 运行相似度评估")
            result_df = self.model.evaluate_dataset(df_filtered)
            
            # 6. 保存结果
            logger.info("步骤 6: 保存结果")
            self._save_results(result_df)
            
            # 7. 保存性能指标
            logger.info("步骤 7: 保存性能指标")
            self.model.save_metrics()
            
            # 8. 打印统计信息
            logger.info("步骤 8: 打印统计信息")
            self._print_final_statistics(result_df)
            
            logger.info("评估管道运行完成！")
            
        except Exception as e:
            logger.error(f"评估管道运行失败: {e}")
            raise
    
    def _save_results(self, result_df):
        """保存结果"""
        # 确保输出目录存在
        output_dir = os.path.dirname(self.config.data.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存结果
        result_df.to_csv(self.config.data.output_csv, index=False)
        logger.info(f"结果已保存到: {self.config.data.output_csv}")
        
        # 保存统计信息
        stats_file = self.config.data.output_csv.replace('.csv', '_stats.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("相似度评估统计信息\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"总记录数: {len(result_df)}\n")
            f.write(f"成功处理: {len(result_df[result_df['status'] == 'success'])}\n")
            f.write(f"处理失败: {len(result_df[result_df['status'] != 'success'])}\n")
            
            if 'similarity_score' in result_df.columns:
                scores = result_df['similarity_score']
                f.write(f"\n相似度分数统计:\n")
                f.write(f"  平均分数: {scores.mean():.4f}\n")
                f.write(f"  最高分数: {scores.max():.4f}\n")
                f.write(f"  最低分数: {scores.min():.4f}\n")
                f.write(f"  标准差: {scores.std():.4f}\n")
                
                # 质量分级
                excellent = len(scores[scores >= 0.8])
                good = len(scores[(scores >= 0.6) & (scores < 0.8)])
                fair = len(scores[(scores >= 0.4) & (scores < 0.6)])
                poor = len(scores[scores < 0.4])
                
                f.write(f"\n质量分级:\n")
                f.write(f"  优秀 (≥0.8): {excellent} ({excellent/len(scores)*100:.1f}%)\n")
                f.write(f"  良好 (0.6-0.8): {good} ({good/len(scores)*100:.1f}%)\n")
                f.write(f"  一般 (0.4-0.6): {fair} ({fair/len(scores)*100:.1f}%)\n")
                f.write(f"  较差 (<0.4): {poor} ({poor/len(scores)*100:.1f}%)\n")
        
        logger.info(f"统计信息已保存到: {stats_file}")
    
    def _print_final_statistics(self, result_df):
        """打印最终统计信息"""
        print("\n" + "="*60)
        print("相似度评估完成统计")
        print("="*60)
        
        print(f"总记录数: {len(result_df)}")
        print(f"成功处理: {len(result_df[result_df['status'] == 'success'])}")
        print(f"处理失败: {len(result_df[result_df['status'] != 'success'])}")
        
        if 'similarity_score' in result_df.columns:
            scores = result_df['similarity_score']
            print(f"\n相似度分数统计:")
            print(f"  平均分数: {scores.mean():.4f}")
            print(f"  最高分数: {scores.max():.4f}")
            print(f"  最低分数: {scores.min():.4f}")
            print(f"  标准差: {scores.std():.4f}")
            
            # 质量分级
            excellent = len(scores[scores >= 0.8])
            good = len(scores[(scores >= 0.6) & (scores < 0.8)])
            fair = len(scores[(scores >= 0.4) & (scores < 0.6)])
            poor = len(scores[scores < 0.4])
            
            print(f"\n质量分级:")
            print(f"  优秀 (≥0.8): {excellent} ({excellent/len(scores)*100:.1f}%)")
            print(f"  良好 (0.6-0.8): {good} ({good/len(scores)*100:.1f}%)")
            print(f"  一般 (0.4-0.6): {fair} ({fair/len(scores)*100:.1f}%)")
            print(f"  较差 (<0.4): {poor} ({poor/len(scores)*100:.1f}%)")
        
        print(f"\n结果文件: {self.config.data.output_csv}")
        print(f"性能指标: performance_metrics.txt")
        print(f"日志文件: {self.config.logging.log_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像-文本相似度评估工具')
    parser.add_argument('--config', '-c', default='config.json', help='配置文件路径')
    parser.add_argument('--input', '-i', help='输入CSV文件路径')
    parser.add_argument('--output', '-o', help='输出CSV文件路径')
    parser.add_argument('--model', '-m', choices=['CLIP', 'BLIP'], help='模型名称 (CLIP/BLIP)')
    parser.add_argument('--device', '-d', help='计算设备')
    parser.add_argument('--batch-size', '-b', type=int, help='批处理大小')
    parser.add_argument('--no-download', action='store_true', help='跳过图像下载')
    parser.add_argument('--no-validate', action='store_true', help='跳过图像验证')
    parser.add_argument('--create-config', action='store_true', help='创建默认配置文件')
    parser.add_argument('--list-models', action='store_true', help='列出支持的模型')
    
    args = parser.parse_args()
    
    try:
        if args.create_config:
            create_config_file(args.config)
            print(f"默认配置文件已创建: {args.config}")
            return
        
        if args.list_models:
            print("支持的模型:")
            print("  CLIP - OpenAI CLIP模型 (推荐用于通用图像-文本相似度)")
            print("    - openai/clip-vit-base-patch32 (默认)")
            print("    - openai/clip-vit-large-patch14")
            print("  BLIP - Salesforce BLIP模型 (推荐用于图像-文本检索)")
            print("    - Salesforce/blip-itm-base-coco (默认)")
            print("    - Salesforce/blip-itm-large-coco")
            return
        
        # 创建评估管道
        pipeline = SimilarityEvaluationPipeline(args.config, args.model)
        
        # 应用命令行参数覆盖
        if args.input:
            pipeline.config.data.input_csv = args.input
        if args.output:
            pipeline.config.data.output_csv = args.output
        if args.device:
            pipeline.config.model.device = args.device
        if args.batch_size:
            pipeline.config.model.batch_size = args.batch_size
        if args.no_download:
            pipeline.config.data.download_images = False
        if args.no_validate:
            pipeline.config.data.validate_images = False
        
        # 运行评估
        pipeline.run()
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
