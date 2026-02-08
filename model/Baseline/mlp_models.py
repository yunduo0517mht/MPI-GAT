"""
MLP Baseline Models for Ablation Study

阶段1: Baseline_MLP - 只使用新闻文本
阶段2: Comment_MLP - 使用新闻文本 + 评论聚合特征（TODO，等数据准备好）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline_MLP(nn.Module):
    """
    阶段1: 只使用新闻文本的MLP基线

    输入: 新闻 BERT embedding (1024维)
    输出: 6类分类

    论文表述: "We establish a weak baseline using only news text with BERT
    embeddings and a simple MLP classifier."
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dims: list = [512, 256],
        num_classes: int = 6,
        dropout_rate: float = 0.5
    ):
        super(Baseline_MLP, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # 构建MLP层
        layers = []
        input_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, news_embedding):
        """
        前向传播

        Args:
            news_embedding: (batch_size, embedding_dim) 或 (embedding_dim,)

        Returns:
            logits: (batch_size, num_classes) 或 (num_classes,)
        """
        # 确保输入是2D张量
        if news_embedding.dim() == 1:
            news_embedding = news_embedding.unsqueeze(0)

        logits = self.mlp(news_embedding)
        return logits

    def predict(self, news_embedding):
        """
        预测（用于推理）

        Args:
            news_embedding: (batch_size, embedding_dim)

        Returns:
            predictions: (batch_size,) 类别索引
            probabilities: (batch_size, num_classes) 概率分布
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(news_embedding)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return predictions, probabilities


class Comment_MLP(nn.Module):
    """
    阶段2: 使用新闻文本 + 评论聚合特征的MLP

    输入:
        - news_embedding: 新闻 BERT embedding (1024维)
        - comment_aggregated: 评论聚合特征 (1024维)

    输出: 6类分类

    TODO: 等评论网络数据准备好后实现训练脚本
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dims: list = [512, 256],
        num_classes: int = 6,
        dropout_rate: float = 0.5
    ):
        super(Comment_MLP, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # 输入维度 = 新闻 + 评论聚合
        input_dim = embedding_dim * 2

        # 构建MLP层
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, news_embedding, comment_aggregated):
        """
        前向传播

        Args:
            news_embedding: (batch_size, embedding_dim)
            comment_aggregated: (batch_size, embedding_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        # 确保输入是2D张量
        if news_embedding.dim() == 1:
            news_embedding = news_embedding.unsqueeze(0)
        if comment_aggregated.dim() == 1:
            comment_aggregated = comment_aggregated.unsqueeze(0)

        # 拼接新闻和评论特征
        combined = torch.cat([news_embedding, comment_aggregated], dim=1)

        logits = self.mlp(combined)
        return logits

    def predict(self, news_embedding, comment_aggregated):
        """
        预测（用于推理）
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(news_embedding, comment_aggregated)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return predictions, probabilities


if __name__ == "__main__":
    """测试模型"""
    print("=" * 60)
    print("Testing Baseline Models")
    print("=" * 60)

    # 测试参数
    embedding_dim = 1024
    batch_size = 4
    num_classes = 6

    # ========== 测试 Baseline_MLP ==========
    print("\n1. Testing Baseline_MLP")
    print("-" * 60)

    model1 = Baseline_MLP(
        embedding_dim=embedding_dim,
        hidden_dims=[512, 256],
        num_classes=num_classes,
        dropout_rate=0.5
    )

    print(f"Model created!")
    print(f"  Total parameters: {sum(p.numel() for p in model1.parameters()):,}")

    # 测试 batch 输入
    news_emb = torch.randn(batch_size, embedding_dim)
    print(f"\nInput (batch): {news_emb.shape}")

    logits1 = model1(news_emb)
    print(f"Output (logits): {logits1.shape}")

    # 测试预测
    predictions, probabilities = model1.predict(news_emb)
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities.shape}")

    # 测试单个样本
    print(f"\n--- Testing single sample ---")
    single_news = torch.randn(embedding_dim)
    logits_single = model1(single_news)
    print(f"Single sample output: {logits_single.shape}")

    # ========== 测试 Comment_MLP ==========
    print("\n2. Testing Comment_MLP")
    print("-" * 60)

    model2 = Comment_MLP(
        embedding_dim=embedding_dim,
        hidden_dims=[512, 256],
        num_classes=num_classes,
        dropout_rate=0.5
    )

    print(f"Model created!")
    print(f"  Total parameters: {sum(p.numel() for p in model2.parameters()):,}")

    # 测试输入
    news_emb = torch.randn(batch_size, embedding_dim)
    comment_emb = torch.randn(batch_size, embedding_dim)
    print(f"\nInput:")
    print(f"  news_embedding: {news_emb.shape}")
    print(f"  comment_aggregated: {comment_emb.shape}")

    logits2 = model2(news_emb, comment_emb)
    print(f"Output (logits): {logits2.shape}")

    # 测试预测
    predictions, probabilities = model2.predict(news_emb, comment_emb)
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities.shape}")

    print("\n" + "=" * 60)
    print("[PASS] All models tested successfully!")
    print("=" * 60)
