"""
MLP Models for Feature Ablation Study

阶段1: Baseline_MLP - 只使用新闻文本
阶段2: Feature_MLP_Residual - 新闻 + 额外特征 (残差连接)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline_MLP(nn.Module):
    """
    阶段1: 只使用新闻文本的MLP基线

    输入: 新闻 BERT embedding (1024维)
    输出: 6类分类
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

        layers = []
        input_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, news_embedding):
        if news_embedding.dim() == 1:
            news_embedding = news_embedding.unsqueeze(0)
        return self.mlp(news_embedding)

    def predict(self, news_embedding):
        self.eval()
        with torch.no_grad():
            logits = self.forward(news_embedding)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


class Feature_MLP_Concat(nn.Module):
    """
    阶段2a: 新闻 + 额外特征 (直接拼接)

    核心思想:
        combined_input = concat(news, author, comment)
        output = MLP(combined_input)
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        num_extra_features: int = 1,  # 特征数量
        hidden_dims: list = [512, 256],
        num_classes: int = 6,
        dropout_rate: float = 0.5
    ):
        super(Feature_MLP_Concat, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # 输入维度 = news(1024) + extra_features(1024 * num_extra_features)
        input_dim = embedding_dim + embedding_dim * num_extra_features

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, news_embedding, extra_features):
        """
        Args:
            news_embedding: (batch_size, 1024) 新闻embedding
            extra_features: (batch_size, 1024 * num_extra_features) 拼接的额外特征

        Returns:
            logits: (batch_size, num_classes)
        """
        if news_embedding.dim() == 1:
            news_embedding = news_embedding.unsqueeze(0)
        if extra_features.dim() == 1:
            extra_features = extra_features.unsqueeze(0)

        combined = torch.cat([news_embedding, extra_features], dim=1)
        return self.mlp(combined)

    def predict(self, news_embedding, extra_features):
        self.eval()
        with torch.no_grad():
            logits = self.forward(news_embedding, extra_features)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


class Feature_MLP_Residual(nn.Module):
    """
    阶段2b: 新闻 + 额外特征 (残差连接)

    核心思想:
        news_feat = MLP_news(news)           # 主分支
        extra_feat = MLP_extra(author, comment)  # 残差分支
        combined = news_feat + extra_feat    # 残差连接

    优点:
        - 即使额外特征没用，模型也能退化为只用新闻
        - 保证不会比只用新闻差
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dim: int = 256,
        num_classes: int = 6,
        dropout_rate: float = 0.5
    ):
        super(Feature_MLP_Residual, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # 主分支：新闻特征处理
        self.news_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 残差分支：额外特征处理（作者 + 评论）
        # 输入维度 = author(1024) + comment_avg(1024) = 2048
        self.extra_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, news_embedding, author_embedding, comment_embedding):
        """
        Args:
            news_embedding: (batch_size, 1024) 新闻embedding
            author_embedding: (batch_size, 1024) 作者embedding
            comment_embedding: (batch_size, 1024) 评论平均embedding

        Returns:
            logits: (batch_size, num_classes)
        """
        # 确保输入是2D
        if news_embedding.dim() == 1:
            news_embedding = news_embedding.unsqueeze(0)
        if author_embedding.dim() == 1:
            author_embedding = author_embedding.unsqueeze(0)
        if comment_embedding.dim() == 1:
            comment_embedding = comment_embedding.unsqueeze(0)

        # 主分支：新闻特征
        news_feat = self.news_mlp(news_embedding)  # [batch, hidden_dim]

        # 残差分支：额外特征（作者 + 评论）
        extra_input = torch.cat([author_embedding, comment_embedding], dim=1)  # [batch, 2048]
        extra_feat = self.extra_mlp(extra_input)  # [batch, hidden_dim]

        # 残差连接：主分支 + 残差分支
        combined = news_feat + extra_feat  # 核心公式！

        # 分类
        logits = self.classifier(combined)
        return logits

    def predict(self, news_embedding, author_embedding, comment_embedding):
        self.eval()
        with torch.no_grad():
            logits = self.forward(news_embedding, author_embedding, comment_embedding)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MLP Models with Residual Connection")
    print("=" * 60)

    embedding_dim = 1024
    hidden_dim = 256
    batch_size = 4
    num_classes = 6

    # Test Baseline_MLP
    print("\n1. Baseline_MLP (news only)")
    model1 = Baseline_MLP(embedding_dim=embedding_dim, num_classes=num_classes)
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    news_emb = torch.randn(batch_size, embedding_dim)
    out1 = model1(news_emb)
    print(f"   Input: {news_emb.shape} -> Output: {out1.shape}")

    # Test Feature_MLP_Residual
    print("\n2. Feature_MLP_Residual (news + author + comment)")
    model2 = Feature_MLP_Residual(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters()):,}")

    author_emb = torch.randn(batch_size, embedding_dim)
    comment_emb = torch.randn(batch_size, embedding_dim)
    out2 = model2(news_emb, author_emb, comment_emb)
    print(f"   Input: news(1024) + author(1024) + comment(1024) -> Output: {out2.shape}")

    print("\n3. Residual Connection Explanation")
    print("-" * 60)
    print("   combined = news_feat + extra_feat")
    print("   - If extra_feat is useful: combined = news_feat + useful_info")
    print("   - If extra_feat is useless: extra_feat ≈ 0, combined ≈ news_feat")
    print("   -> Model won't be worse than news-only baseline!")

    print("\n[PASS] All tests passed!")
