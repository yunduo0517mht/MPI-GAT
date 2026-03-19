"""
GCN/GAT 模型实现

用于新闻-评论二部图的假新闻检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class CommentGCN(nn.Module):
    """
    GCN 模型：评论 → 新闻 单向聚合

    架构：
    1. 评论特征通过 GCN 聚合到新闻节点
    2. 残差连接：最终特征 = 原始新闻特征 + 聚合的评论特征
    3. MLP 分类器
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 6,
        dropout_rate: float = 0.6
    ):
        super(CommentGCN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # GCN 层：用于聚合评论特征
        # 这里我们使用简单的线性变换 + 归一化聚合
        self.comment_transform = nn.Linear(embedding_dim, embedding_dim)

        # 可学习的聚合权重（类似于 GCN 的度归一化）
        self.aggregation_weight = nn.Parameter(torch.ones(1))

        # 残差门控
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )

        # MLP 分类器
        layers = []
        input_dim = embedding_dim  # 残差融合后保持原始维度

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        self.classifier = nn.Sequential(*layers)

    def aggregate_comments(
        self,
        news_feat: torch.Tensor,
        comment_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        聚合评论特征到新闻节点（GCN 风格）

        Args:
            news_feat: (embedding_dim,) 新闻特征
            comment_feats: (num_comments, embedding_dim) 评论特征

        Returns:
            aggregated: (embedding_dim,) 聚合后的特征
        """
        num_comments = comment_feats.size(0)

        # 线性变换评论特征
        comment_transformed = self.comment_transform(comment_feats)  # (num_comments, embedding_dim)

        # GCN 风格的归一化聚合 (1/sqrt(N) * sum)
        # 这里使用可学习的权重
        norm_factor = 1.0 / (num_comments + 1)  # +1 是新闻节点自身

        # 平均聚合
        aggregated = torch.mean(comment_transformed, dim=0)  # (embedding_dim,)

        return aggregated

    def forward(
        self,
        news_feats: torch.Tensor,
        comment_feats_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播

        Args:
            news_feats: (batch_size, embedding_dim) 新闻特征
            comment_feats_list: List of (num_comments, embedding_dim)
                每个样本的评论特征

        Returns:
            logits: (batch_size, num_classes)
            debug_info: Dict 调试信息
        """
        batch_size = news_feats.size(0)
        device = news_feats.device

        # 聚合每个样本的评论特征
        aggregated_list = []
        for i in range(batch_size):
            comment_feats = comment_feats_list[i].to(device)

            if comment_feats.size(0) == 0:
                # 没有评论，使用零向量
                aggregated = torch.zeros(self.embedding_dim, device=device)
            else:
                aggregated = self.aggregate_comments(news_feats[i], comment_feats)

            aggregated_list.append(aggregated)

        # 堆叠成 batch
        aggregated_batch = torch.stack(aggregated_list, dim=0)  # (batch_size, embedding_dim)

        # 残差门控融合
        gate_input = torch.cat([news_feats, aggregated_batch], dim=-1)
        gate = self.gate(gate_input)  # (batch_size, embedding_dim)

        # 最终特征 = 原始新闻特征 + 门控 * 聚合特征
        fused_feat = news_feats + gate * aggregated_batch

        # 分类
        logits = self.classifier(fused_feat)

        debug_info = {
            'aggregated': aggregated_batch,
            'gate': gate,
            'fused_feat': fused_feat
        }

        return logits, debug_info


class CommentGAT(nn.Module):
    """
    GAT 模型：注意力聚合评论特征

    架构：
    1. 评论特征通过注意力机制聚合到新闻节点
    2. 残差连接：最终特征 = 原始新闻特征 + 聚合的评论特征
    3. MLP 分类器
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 6,
        heads: int = 1,
        dropout_rate: float = 0.6
    ):
        super(CommentGAT, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.heads = heads

        # 注意力层
        # 注意力计算：a(Wh_i || Wh_j)
        self.W_news = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_comment = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attention = nn.Parameter(torch.zeros(2 * embedding_dim, 1))

        nn.init.xavier_uniform_(self.attention.data, gain=1.414)

        # 残差门控
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )

        # MLP 分类器
        layers = []
        input_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        self.classifier = nn.Sequential(*layers)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def compute_attention(
        self,
        news_feat: torch.Tensor,
        comment_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算注意力权重并聚合

        Args:
            news_feat: (embedding_dim,) 新闻特征
            comment_feats: (num_comments, embedding_dim) 评论特征

        Returns:
            aggregated: (embedding_dim,) 聚合后的特征
            attention_weights: (num_comments,) 注意力权重
        """
        num_comments = comment_feats.size(0)

        # 线性变换
        h_news = self.W_news(news_feat.unsqueeze(0))  # (1, embedding_dim)
        h_comments = self.W_comment(comment_feats)    # (num_comments, embedding_dim)

        # 扩展新闻特征以匹配评论数量
        h_news_expanded = h_news.expand(num_comments, -1)  # (num_comments, embedding_dim)

        # 拼接计算注意力分数
        concat_features = torch.cat([h_news_expanded, h_comments], dim=-1)  # (num_comments, 2*embedding_dim)
        attention_scores = self.leaky_relu(torch.matmul(concat_features, self.attention)).squeeze(-1)  # (num_comments,)

        # Softmax 归一化
        attention_weights = F.softmax(attention_scores, dim=0)  # (num_comments,)

        # 加权聚合
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * h_comments, dim=0)  # (embedding_dim,)

        return aggregated, attention_weights

    def forward(
        self,
        news_feats: torch.Tensor,
        comment_feats_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播

        Args:
            news_feats: (batch_size, embedding_dim) 新闻特征
            comment_feats_list: List of (num_comments, embedding_dim)
                每个样本的评论特征

        Returns:
            logits: (batch_size, num_classes)
            debug_info: Dict 调试信息
        """
        batch_size = news_feats.size(0)
        device = news_feats.device

        # 聚合每个样本的评论特征
        aggregated_list = []
        attention_weights_list = []

        for i in range(batch_size):
            comment_feats = comment_feats_list[i].to(device)

            if comment_feats.size(0) == 0:
                # 没有评论，使用零向量
                aggregated = torch.zeros(self.embedding_dim, device=device)
                attention_weights = torch.tensor([], device=device)
            else:
                aggregated, attention_weights = self.compute_attention(
                    news_feats[i], comment_feats
                )

            aggregated_list.append(aggregated)
            attention_weights_list.append(attention_weights)

        # 堆叠成 batch
        aggregated_batch = torch.stack(aggregated_list, dim=0)  # (batch_size, embedding_dim)

        # 残差门控融合
        gate_input = torch.cat([news_feats, aggregated_batch], dim=-1)
        gate = self.gate(gate_input)  # (batch_size, embedding_dim)

        # 最终特征 = 原始新闻特征 + 门控 * 聚合特征
        fused_feat = news_feats + gate * aggregated_batch

        # 分类
        logits = self.classifier(fused_feat)

        debug_info = {
            'aggregated': aggregated_batch,
            'gate': gate,
            'fused_feat': fused_feat,
            'attention_weights': attention_weights_list
        }

        return logits, debug_info


class AvgPoolBaseline(nn.Module):
    """
    基线模型：简单平均聚合评论特征

    用于对比，验证 GNN 聚合是否比简单平均更有效
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 6,
        dropout_rate: float = 0.6
    ):
        super(AvgPoolBaseline, self).__init__()

        self.embedding_dim = embedding_dim

        # 残差门控
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )

        # MLP 分类器
        layers = []
        input_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        news_feats: torch.Tensor,
        comment_feats_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播

        Args:
            news_feats: (batch_size, embedding_dim) 新闻特征
            comment_feats_list: List of (num_comments, embedding_dim)

        Returns:
            logits: (batch_size, num_classes)
            debug_info: Dict
        """
        batch_size = news_feats.size(0)
        device = news_feats.device

        # 简单平均聚合
        aggregated_list = []
        for i in range(batch_size):
            comment_feats = comment_feats_list[i].to(device)

            if comment_feats.size(0) == 0:
                aggregated = torch.zeros(self.embedding_dim, device=device)
            else:
                aggregated = torch.mean(comment_feats, dim=0)

            aggregated_list.append(aggregated)

        aggregated_batch = torch.stack(aggregated_list, dim=0)

        # 残差门控融合
        gate_input = torch.cat([news_feats, aggregated_batch], dim=-1)
        gate = self.gate(gate_input)

        fused_feat = news_feats + gate * aggregated_batch

        # 分类
        logits = self.classifier(fused_feat)

        debug_info = {
            'aggregated': aggregated_batch,
            'gate': gate
        }

        return logits, debug_info


if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("Testing GNN Models")
    print("=" * 60)

    embedding_dim = 1024
    batch_size = 4

    # 创建模型
    gcn = CommentGCN(embedding_dim=embedding_dim, hidden_dims=[512, 256], num_classes=6)
    gat = CommentGAT(embedding_dim=embedding_dim, hidden_dims=[512, 256], num_classes=6, heads=1)
    avg_pool = AvgPoolBaseline(embedding_dim=embedding_dim, hidden_dims=[512, 256], num_classes=6)

    print(f"\nModel parameters:")
    print(f"  GCN: {sum(p.numel() for p in gcn.parameters()):,}")
    print(f"  GAT: {sum(p.numel() for p in gat.parameters()):,}")
    print(f"  AvgPool: {sum(p.numel() for p in avg_pool.parameters()):,}")

    # 模拟数据
    news_feats = torch.randn(batch_size, embedding_dim)
    comment_feats_list = [
        torch.randn(6, embedding_dim),
        torch.randn(8, embedding_dim),
        torch.randn(3, embedding_dim),
        torch.randn(5, embedding_dim),
    ]

    # 测试 GCN
    print("\n--- Testing GCN ---")
    logits_gcn, debug_gcn = gcn(news_feats, comment_feats_list)
    print(f"  logits: {logits_gcn.shape}")
    print(f"  gate mean: {debug_gcn['gate'].mean().item():.4f}")

    # 测试 GAT
    print("\n--- Testing GAT ---")
    logits_gat, debug_gat = gat(news_feats, comment_feats_list)
    print(f"  logits: {logits_gat.shape}")
    print(f"  gate mean: {debug_gat['gate'].mean().item():.4f}")
    print(f"  attention weights (sample 0): {debug_gat['attention_weights'][0]}")

    # 测试 AvgPool
    print("\n--- Testing AvgPool ---")
    logits_avg, debug_avg = avg_pool(news_feats, comment_feats_list)
    print(f"  logits: {logits_avg.shape}")
    print(f"  gate mean: {debug_avg['gate'].mean().item():.4f}")

    print("\n[PASS] Model tests passed!")
