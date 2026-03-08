"""
MPI-GAT: Multi-Path Interactive Graph Attention Network
简化版主模型（只使用 NAN 元路径）

整体架构：
1. 输入：新闻 embedding (1024维)
2. 提取 NAN 元路径特征
3. 拼接所有特征（原始 + NAN）
4. MLP 降维
5. 分类器输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from .metapath_modules import NANMetapath


class MPIGAT_Simple(nn.Module):
    """
    MPI-GAT 简化版
    只使用 NAN (News-Author-News) 元路径
    """
    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 6,
        nan_k: int = 3,
        dropout_rate: float = 0.5
    ):
        super(MPIGAT_Simple, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # ===== NAN 元路径模块 =====
        self.nan_module = NANMetapath(
            embedding_dim=embedding_dim,
            k=nan_k,
            dropout_rate=0.3
        )

        # ===== NAN 门控网络 =====
        self.nan_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, 1)
        )

        # ===== 融合和降维 MLP =====
        # 输入：原始特征(1024) + NAN特征(1024) = 2048
        fusion_input_dim = embedding_dim * 2

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ===== 分类器 =====
        self.classifier = nn.Linear(hidden_dims[1], num_classes)

    def forward(
        self,
        news_embeddings: torch.Tensor,
        author_news_embeddings_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            news_embeddings: (batch_size, embedding_dim)
            author_news_embeddings_list: List of (num_news, embedding_dim)
                每个样本的作者新闻列表（长度可能不同）

        Returns:
            logits: (batch_size, num_classes)
            debug_info: Dict 调试信息
        """
        batch_size = news_embeddings.size(0)
        device = news_embeddings.device

        # ===== 步骤1：提取 NAN 元路径特征 =====
        z_nan_list = []
        u_a_list = []
        top_k_scores_list = []

        for i in range(batch_size):
            z_nan, u_a, top_k_scores, debug_info = self.nan_module(
                news_embeddings[i],
                author_news_embeddings_list[i]
            )
            z_nan_list.append(z_nan)
            u_a_list.append(u_a)
            top_k_scores_list.append(top_k_scores)

        z_nan_batch = torch.stack(z_nan_list, dim=0)  # (batch_size, embedding_dim)
        u_a_batch = torch.stack(u_a_list, dim=0)  # (batch_size, embedding_dim)

        # ===== 步骤2：对 NAN 特征做门控 =====
        gate = torch.sigmoid(self.nan_gate(news_embeddings))  # (batch_size, 1)
        z_nan_gated = gate * z_nan_batch  # (batch_size, embedding_dim)

        # ===== 步骤3：拼接所有特征 =====
        all_features = torch.cat([
            news_embeddings,
            z_nan_gated
        ], dim=1)  # (batch_size, 2*embedding_dim)

        # ===== 步骤4：MLP 降维 =====
        hidden = self.fusion_mlp(all_features)  # (batch_size, hidden_dims[-1])

        # ===== 步骤5：分类 =====
        logits = self.classifier(hidden)  # (batch_size, num_classes)

        debug_info = {
            'z_nan': z_nan_batch,
            'u_A': u_a_batch,
            'gate': gate,
            'top_k_scores': top_k_scores_list
        }

        return logits, debug_info


if __name__ == "__main__":
    # 测试主模型
    print("=" * 60)
    print("Testing MPI-GAT Simple Model")
    print("=" * 60)

    # 模型参数
    embedding_dim = 1024
    batch_size = 4

    # 创建模型
    model = MPIGAT_Simple(
        embedding_dim=embedding_dim,
        hidden_dims=[512, 256],
        num_classes=6,
        nan_k=3,
        dropout_rate=0.5
    )

    print(f"\nModel initialized!")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟数据（每个样本的作者新闻数量不同）
    news_emb = torch.randn(batch_size, embedding_dim)
    author_news_emb_list = [
        torch.randn(1, embedding_dim),   # 样本0: 1篇
        torch.randn(5, embedding_dim),   # 样本1: 5篇
        torch.randn(3, embedding_dim),   # 样本2: 3篇
        torch.randn(10, embedding_dim),  # 样本3: 10篇
    ]

    print(f"\nInput:")
    print(f"  news_embeddings: {news_emb.shape}")
    print(f"  author_news_embeddings_list: {len(author_news_emb_list)} samples")
    for i, emb in enumerate(author_news_emb_list):
        print(f"    Sample {i}: {emb.shape}")

    # 前向传播
    logits, debug_info = model(news_emb, author_news_emb_list)

    print(f"\nOutput:")
    print(f"  logits: {logits.shape}")
    print(f"  z_nan: {debug_info['z_nan'].shape}")
    print(f"  u_A: {debug_info['u_A'].shape}")
    print(f"  gate: {debug_info['gate'].shape}")

    print("\n[PASS] MPI-GAT Simple model test passed!")
