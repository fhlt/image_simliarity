"""
Models module for image-text similarity evaluation
包含各种相似度评估模型的实现
"""

from .clip_evaluator import CLIPSimilarityEvaluator
from .blip_evaluator import BLIPSimilarityEvaluator
from .base_evaluator import BaseSimilarityEvaluator

__all__ = ['CLIPSimilarityEvaluator', 'BLIPSimilarityEvaluator', 'BaseSimilarityEvaluator']
