"""
MPI-GAT: Multi-Path Interactive Graph Attention Network
元路径模块实现

包含5个元路径的提取和融合：
1. NAN: News-Author-News (作者写作意图) ✓ 已实现（同构图版本）
2. NCN: News-Comment-News (评论证据) - TODO
3. NCA: News-Comment-Author (读者反馈) - TODO
4. CAC: Comment-Author-Comment (作者-读者互动) - TODO
5. CCC: Comment-Comment-Comment (读者间互动) - TODO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class NewsImportanceScorer(nn.Module):
    """
    新闻重要性评分网络
    评估每篇新闻对作者写作意图的贡献度

    输入：[新闻embedding + 作者平均embedding]
    输出：重要性分数 (0-1之间)
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = None):
        super(NewsImportanceScorer, self).__init__()
        if hidden_dim is None:
            hidden_dim = embedding_dim // 4

        self.embedding_dim = embedding_dim

        # 重要性评分网络
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        news_embedding: torch.Tensor,
        author_avg_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            news_embedding: (num_news, embedding_dim) 或 (embedding_dim,)
            author_avg_embedding: (embedding_dim,) 作者的平均风格

        Returns:
            importance_scores: (num_news,) 归一化后的重要性分数
        """
        # 处理单个样本的情况
        squeeze_output = False
        if news_embedding.dim() == 1:
            news_embedding = news_embedding.unsqueeze(0)
            squeeze_output = True

        num_news = news_embedding.size(0)

        # 拼接新闻和作者平均
        author_avg_expanded = author_avg_embedding.unsqueeze(0).expand(num_news, -1)
        combined = torch.cat([news_embedding, author_avg_expanded], dim=1)  # (num_news, 2*embedding_dim)

        # 计算重要性分数
        raw_scores = self.scorer(combined)  # (num_news, 1)
        importance_scores = F.softmax(raw_scores.squeeze(-1), dim=0)  # (num_news,)

        if squeeze_output:
            importance_scores = importance_scores.squeeze(0)

        return importance_scores


class EdgeWeightComputer(nn.Module):
    """
    边权重计算网络
    计算同作者新闻之间边的权重

    输入：[新闻i + 新闻j]
    输出：边权重 (0-1之间)
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = None):
        super(EdgeWeightComputer, self).__init__()
        if hidden_dim is None:
            hidden_dim = embedding_dim // 4

        # 边权重网络
        self.computer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1之间
        )

    def forward(
        self,
        news_i_embeddings: torch.Tensor,
        news_j_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            news_i_embeddings: (num_edges, embedding_dim)
            news_j_embeddings: (num_edges, embedding_dim)

        Returns:
            edge_weights: (num_edges,) 边权重
        """
        # 拼接两端新闻
        combined = torch.cat([news_i_embeddings, news_j_embeddings], dim=1)  # (num_edges, 2*embedding_dim)

        # 计算边权重
        edge_weights = self.computer(combined).squeeze(-1)  # (num_edges,)

        return edge_weights


class GraphAttentionPooling(nn.Module):
    """
    图上的 Attention Pooling
    对选中的新闻做加权聚合
    """
    def __init__(self, embedding_dim: int):
        super(GraphAttentionPooling, self).__init__()
        self.embedding_dim = embedding_dim

        # Attention 参数
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(
        self,
        news_embeddings: torch.Tensor,
        importance_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            news_embeddings: (num_news, embedding_dim)
            importance_scores: (num_news,) 重要性分数

        Returns:
            pooled_embedding: (embedding_dim,) 聚合后的向量
        """
        num_news = news_embeddings.size(0)

        # 计算注意力权重（结合重要性分数）
        attn_scores = self.attention(news_embeddings).squeeze(-1)  # (num_news,)

        # 结合重要性分数和注意力
        combined_scores = F.softmax(attn_scores + torch.log(importance_scores + 1e-8), dim=0)

        # 加权聚合
        pooled = torch.sum(combined_scores.unsqueeze(-1) * news_embeddings, dim=0)  # (embedding_dim,)

        return pooled


class NANMetapath(nn.Module):
    """
    NAN: News-Author-News 元路径（同构图版本）

    设计思路：
    1. 构建虚拟中心节点，连接该作者的所有新闻
    2. 计算每篇新闻的重要性分数（基于内容 + 作者平均风格）
    3. 选择 Top-k 新闻
    4. 计算 Top-k 子图上的边权重
    5. GNN 消息传递（单层）
    6. Attention Pooling 提取作者写作意图 u_A
    7. 门控融合：z_NAN = a_N + gate * W_u * u_A
    """
    def __init__(
        self,
        embedding_dim: int,
        k: int = 3,
        dropout_rate: float = 0.3
    ):
        super(NANMetapath, self).__init__()
        self.embedding_dim = embedding_dim
        self.k = k

        # 重要性评分器
        self.importance_scorer = NewsImportanceScorer(embedding_dim)

        # 边权重计算器
        self.edge_weight_computer = EdgeWeightComputer(embedding_dim)

        # GNN 层（用于消息传递）
        self.gnn_layer = nn.Linear(embedding_dim, embedding_dim)

        # Attention Pooling
        self.attention_pooling = GraphAttentionPooling(embedding_dim)

        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 特征变换
        self.feature_transform = nn.Linear(embedding_dim, embedding_dim)

        # Dropout（防捷径）
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        news_embedding: torch.Tensor,
        author_news_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            news_embedding: (embedding_dim,) 当前新闻的embedding
            author_news_embeddings: (num_news, embedding_dim)
                该作者的所有新闻embedding（包括当前新闻）

        Returns:
            z_NAN: (embedding_dim,) 增强后的新闻特征
            u_A: (embedding_dim,) 作者写作意图
            top_k_scores: (k,) top-k 新闻的权重
            debug_info: Dict 调试信息
        """
        num_news = author_news_embeddings.size(0)
        device = author_news_embeddings.device

        debug_info = {}

        # ===== 步骤1：计算作者平均风格 =====
        author_avg_embedding = torch.mean(author_news_embeddings, dim=0)  # (embedding_dim,)
        debug_info['author_avg'] = author_avg_embedding

        # ===== 步骤2：计算每篇新闻的重要性分数 =====
        importance_scores = self.importance_scorer(
            author_news_embeddings,
            author_avg_embedding
        )  # (num_news,)
        debug_info['importance_scores'] = importance_scores

        # ===== 步骤3：选择 Top-k =====
        if num_news <= self.k:
            # 如果新闻数 <= k，全部保留
            top_k_indices = torch.arange(num_news, device=device)
        else:
            # 选择重要性最高的 k 个
            top_k_indices = torch.topk(importance_scores, self.k).indices

        top_k_embeddings = author_news_embeddings[top_k_indices]  # (k, embedding_dim)
        top_k_scores = importance_scores[top_k_indices]  # (k,)
        debug_info['top_k_indices'] = top_k_indices
        debug_info['top_k_scores'] = top_k_scores

        # ===== 步骤4：构建 Top-k 子图（虚拟中心节点结构） =====
        # 虚拟中心节点：所有 Top-k 新闻的平均
        virtual_center = torch.mean(top_k_embeddings, dim=0)  # (embedding_dim,)

        # 实际的 k 值（可能小于预设的 k）
        actual_k = top_k_embeddings.size(0)

        # 构建边：虚拟中心 <-> 每篇新闻
        # 双向边，所以有 2*actual_k 条边
        num_edges = 2 * actual_k
        edge_index = torch.zeros((2, num_edges), dtype=torch.long, device=device)
        edge_weights = torch.zeros(num_edges, device=device)

        # 从中心到新闻的边
        center_to_news_indices = torch.arange(actual_k, device=device)
        edge_index[0, :actual_k] = actual_k  # 中心节点索引设为 actual_k
        edge_index[1, :actual_k] = center_to_news_indices

        # 从新闻到中心的边
        edge_index[0, actual_k:] = center_to_news_indices
        edge_index[1, actual_k:] = actual_k  # 中心节点索引设为 actual_k

        # 计算边权重
        center_expanded = virtual_center.unsqueeze(0).expand(actual_k, -1)  # (actual_k, embedding_dim)

        # 中心 -> 新闻
        weights_center_to_news = self.edge_weight_computer(center_expanded, top_k_embeddings)
        edge_weights[:actual_k] = weights_center_to_news

        # 新闻 -> 中心
        weights_news_to_center = self.edge_weight_computer(top_k_embeddings, center_expanded)
        edge_weights[actual_k:] = weights_news_to_center

        debug_info['edge_weights'] = edge_weights

        # ===== 步骤5：GNN 消息传递 =====
        # 构建节点特征（包含虚拟中心节点）
        all_embeddings = torch.cat([top_k_embeddings, virtual_center.unsqueeze(0)], dim=0)  # (actual_k+1, embedding_dim)

        # 消息传递
        # 源节点特征
        source_features = all_embeddings[edge_index[0]]  # (num_edges, embedding_dim)
        # 边权重
        edge_weights_expanded = edge_weights.unsqueeze(-1)  # (num_edges, 1)
        # 加权消息
        messages = source_features * edge_weights_expanded  # (num_edges, embedding_dim)

        # 聚合消息（求和）
        num_nodes = actual_k + 1
        aggregated = torch.zeros(num_nodes, self.embedding_dim, device=device)
        aggregated.scatter_add_(0, edge_index[1].unsqueeze(-1).expand(-1, self.embedding_dim), messages)

        # 应用 GNN 变换
        news_enhanced = self.gnn_layer(aggregated[:actual_k])  # (actual_k, embedding_dim) 只取新闻节点

        # 加上残差连接
        news_enhanced = news_enhanced + top_k_embeddings  # (actual_k, embedding_dim)

        debug_info['news_enhanced'] = news_enhanced

        # ===== 步骤6：Attention Pooling 提取作者意图 =====
        u_A = self.attention_pooling(news_enhanced, top_k_scores)  # (embedding_dim,)
        debug_info['u_A'] = u_A

        # ===== 步骤7：门控融合 =====
        # 对 u_A 做 dropout（防捷径学习）
        u_A_drop = self.dropout(u_A)

        # 特征变换
        u_A_transformed = self.feature_transform(u_A_drop)

        # 计算门控值（向量门控）
        gate_input = torch.cat([news_embedding, u_A_transformed], dim=0)
        gate = torch.sigmoid(self.gate_network(gate_input))  # (embedding_dim,)

        # 残差注入
        z_NAN = news_embedding + gate * u_A_transformed

        return z_NAN, u_A, top_k_scores, debug_info


# TODO: 其他元路径模块（占位符）

class NCNMetapath(nn.Module):
    """
    NCN: News-Comment-News 元路径
    提取评论证据特征

    TODO: 待实现
    """
    def __init__(self, embedding_dim: int, dropout_rate: float = 0.1):
        super(NCNMetapath, self).__init__()
        self.embedding_dim = embedding_dim
        # TODO: 初始化

    def forward(
        self,
        news_embedding: torch.Tensor,
        comment_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # TODO: 实现
        raise NotImplementedError("NCNMetapath not implemented yet")


class NCAMetapath(nn.Module):
    """
    NCA: News-Comment-Author 元路径
    提取读者对作者的反馈特征

    TODO: 待实现
    """
    def __init__(self, embedding_dim: int, dropout_rate: float = 0.1):
        super(NCAMetapath, self).__init__()
        self.embedding_dim = embedding_dim
        # TODO: 初始化

    def forward(
        self,
        news_embedding: torch.Tensor,
        comment_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # TODO: 实现
        raise NotImplementedError("NCAMetapath not implemented yet")


class CACMetapath(nn.Module):
    """
    CAC: Comment-Author-Comment 元路径
    提取作者-读者互动特征

    TODO: 待实现
    """
    def __init__(self, embedding_dim: int, dropout_rate: float = 0.1):
        super(CACMetapath, self).__init__()
        self.embedding_dim = embedding_dim
        # TODO: 初始化

    def forward(
        self,
        news_embedding: torch.Tensor,
        comment_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # TODO: 实现
        raise NotImplementedError("CACMetapath not implemented yet")


class CCCMetapath(nn.Module):
    """
    CCC: Comment-Comment-Comment 元路径
    提取读者间互动特征

    TODO: 待实现
    """
    def __init__(self, embedding_dim: int, dropout_rate: float = 0.1):
        super(CCCMetapath, self).__init__()
        self.embedding_dim = embedding_dim
        # TODO: 初始化

    def forward(
        self,
        news_embedding: torch.Tensor,
        comment_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # TODO: 实现
        raise NotImplementedError("CCCMetapath not implemented yet")


if __name__ == "__main__":
    # 测试 NAN 模块
    print("=" * 60)
    print("Testing NAN Module (Homogeneous Graph Version)")
    print("=" * 60)

    embedding_dim = 1024
    k = 3

    # 创建 NAN 模块
    nan = NANMetapath(embedding_dim, k=k)

    # 模拟数据
    news_emb = torch.randn(embedding_dim)
    author_news_emb = torch.randn(10, embedding_dim)  # 假设作者写了10篇新闻

    print(f"\nInput:")
    print(f"  news_embedding shape: {news_emb.shape}")
    print(f"  author_news_embeddings shape: {author_news_emb.shape}")

    # 前向传播
    z_nan, u_a, top_k_scores, debug_info = nan(news_emb, author_news_emb)

    print(f"\nOutput:")
    print(f"  z_NAN shape: {z_nan.shape}")
    print(f"  u_A shape: {u_a.shape}")
    print(f"  top_k_scores: {top_k_scores}")
    print(f"  top_k_scores sum: {top_k_scores.sum().item():.4f}")
    print(f"\nDebug Info:")
    print(f"  importance_scores (first 5): {debug_info['importance_scores'][:5]}")
    print(f"  top_k_indices: {debug_info['top_k_indices']}")
    print(f"  edge_weights: {debug_info['edge_weights']}")
    print(f"  news_enhanced shape: {debug_info['news_enhanced'].shape}")

    print("\n[PASS] NAN module test passed!")
