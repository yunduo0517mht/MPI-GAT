"""
GNN 模块：基础图神经网络模型

包含：
- CommentGCN: 基于 GCN 的评论聚合模型
- CommentGAT: 基于 GAT 的注意力聚合模型
"""

from .gnn_models import CommentGCN, CommentGAT

__all__ = ['CommentGCN', 'CommentGAT']
