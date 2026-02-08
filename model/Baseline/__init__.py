"""
Baseline models for ablation study

包含两个MLP基线模型：
1. Baseline_MLP: 只使用新闻文本
2. Comment_MLP: 使用新闻文本 + 评论聚合特征
"""

from .mlp_models import Baseline_MLP, Comment_MLP

__all__ = ['Baseline_MLP', 'Comment_MLP']
