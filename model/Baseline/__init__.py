"""
Baseline models for ablation study

包含三个MLP模型：
1. Baseline_MLP: 只使用新闻文本
2. Feature_MLP_Concat: 新闻 + 额外特征 (直接拼接)
3. Feature_MLP_Residual: 新闻 + 额外特征 (残差融合)
"""

from .mlp_models import Baseline_MLP, Feature_MLP_Concat, Feature_MLP_Residual

__all__ = ['Baseline_MLP', 'Feature_MLP_Concat', 'Feature_MLP_Residual']
