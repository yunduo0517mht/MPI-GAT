"""
MPI-GAT: Multi-Path Interactive Graph Attention Network
主模型文件

整体架构：
1. 输入：新闻 embedding (1024维)
2. 提取5个元路径特征
3. 对每个元路径特征做门控
4. 拼接所有特征（原始 + 5个元路径）
5. MLP 降维
6. 分类器输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from metapath_modules import (
    NANMetapath,
    NCNMetapath,
    NCAMetapath,
    CACMetapath,
    CCCMetapath
)


class MPIGAT(nn.Module):
    """
    MPI-GAT 主模型

    多路径交互图注意力网络
    """
    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 6,
        nan_k: int = 3,
        dropout_rate: float = 0.5
    ):
        super(MPIGAT, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # ===== 元路径模块 =====
        # 1. NAN: News-Author-News
        self.nan_module = NANMetapath(
            embedding_dim=embedding_dim,
            k=nan_k,
            dropout_rate=0.3
        )

        # 2. NCN: News-Comment-News (TODO)
        self.ncn_module = None  # TODO: NCNMetapath(embedding_dim)

        # 3. NCA: News-Comment-Author (TODO)
        self.nca_module = None  # TODO: NCAMetapath(embedding_dim)

        # 4. CAC: Comment-Author-Comment (TODO)
        self.cac_module = None  # TODO: CACMetapath(embedding_dim)

        # 5. CCC: Comment-Comment-Comment (TODO)
        self.ccc_module = None  # TODO: CCCMetapath(embedding_dim)

        # ===== 元路径门控网络 =====
        # 对每个元路径特征做独立的门控
        self.metapath_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 4),
                nn.ReLU(),
                nn.Linear(embedding_dim // 4, 1)
            ) for _ in range(5)  # 5个元路径
        ])

        # ===== 融合和降维 MLP =====
        # 输入：原始特征(1024) + 5个元路径特征(5*1024) = 6144
        fusion_input_dim = embedding_dim * 6  # 原始 + 5个元路径

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
        author_news_embeddings_list: List[torch.Tensor],
        comment_embeddings: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            news_embeddings: (batch_size, embedding_dim)
            author_news_embeddings_list: List of (num_news, embedding_dim)
                每个样本的作者新闻列表（长度可能不同）
            comment_embeddings: (可选) 评论特征

        Returns:
            logits: (batch_size, num_classes)
            metapath_features: Dict[str, torch.Tensor] 各个元路径的特征
        """
        batch_size = news_embeddings.size(0)

        # ===== 步骤1：提取 NAN 元路径特征 =====
        z_nan_list = []
        for i in range(batch_size):
            z_nan, u_a, top_k_scores, debug_info = self.nan_module(
                news_embeddings[i],
                author_news_embeddings_list[i]
            )
            z_nan_list.append(z_nan)

        z_nan_batch = torch.stack(z_nan_list, dim=0)  # (batch_size, embedding_dim)

        # 2-5. 其他元路径 (TODO: 占位符)
        # 暂时用零向量代替
        z_ncn_batch = torch.zeros_like(z_nan_batch)
        z_nca_batch = torch.zeros_like(z_nan_batch)
        z_cac_batch = torch.zeros_like(z_nan_batch)
        z_ccc_batch = torch.zeros_like(z_nan_batch)

        metapath_features['z_NCN'] = z_ncn_batch
        metapath_features['z_NCA'] = z_nca_batch
        metapath_features['z_CAC'] = z_cac_batch
        metapath_features['z_CCC'] = z_ccc_batch

        # ===== 步骤2：对每个元路径特征做门控 =====
        metapath_features_gated = {}
        gate_values = {}

        for idx, (name, z) in enumerate(metapath_features.items()):
            # 计算门控值（标量门控）
            gate = torch.sigmoid(self.metapath_gates[idx](news_embedding))
            gate_values[name] = gate

            # 应用门控
            z_gated = gate * z
            metapath_features_gated[name] = z_gated

        # ===== 步骤3：拼接所有特征 =====
        # 原始特征 + 5个元路径特征
        all_features = torch.cat([
            news_embedding,
            metapath_features_gated['z_NAN'],
            metapath_features_gated['z_NCN'],
            metapath_features_gated['z_NCA'],
            metapath_features_gated['z_CAC'],
            metapath_features_gated['z_CCC'],
        ], dim=1)  # (batch_size, 6*embedding_dim)

        # ===== 步骤4：MLP 降维 =====
        hidden = self.fusion_mlp(all_features)  # (batch_size, hidden_dims[-1])

        # ===== 步骤5：分类 =====
        logits = self.classifier(hidden)  # (batch_size, num_classes)

        # 如果输入是单个样本，去掉batch维度
        if squeeze_output:
            logits = logits.squeeze(0)
            for name in metapath_features:
                metapath_features[name] = metapath_features[name].squeeze(0)

        return logits, metapath_features


if __name__ == "__main__":
    # 测试主模型
    print("=" * 60)
    print("Testing MPI-GAT Model")
    print("=" * 60)

    # 模型参数
    embedding_dim = 1024
    batch_size = 4
    num_news_per_author = 10

    # 创建模型
    model = MPIGAT(
        embedding_dim=embedding_dim,
        hidden_dims=[512, 256],
        num_classes=6,
        nan_k=3,
        dropout_rate=0.5
    )

    print(f"\nModel initialized!")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

    # 模拟数据
    news_emb = torch.randn(batch_size, embedding_dim)
    author_news_emb = torch.randn(batch_size, num_news_per_author, embedding_dim)

    print(f"\nInput:")
    print(f"  news_embedding: {news_emb.shape}")
    print(f"  author_news_embeddings: {author_news_emb.shape}")

    # 前向传播
    logits, metapath_features = model(news_emb, author_news_emb)

    print(f"\nOutput:")
    print(f"  logits: {logits.shape}")
    print(f"  metapath_features keys: {list(metapath_features.keys())}")
    for name, feat in metapath_features.items():
        print(f"    {name}: {feat.shape}")

    # 测试单个样本
    print(f"\n--- Testing single sample ---")
    logits_single, features_single = model(
        news_emb[0],
        author_news_emb[0]
    )
    print(f"  Single sample logits: {logits_single.shape}")

    print("\n[PASS] MPI-GAT model test passed!")
