"""
MPI-GAT 数据加载器

从 PyTorch Geometric 的 HeteroData 格式加载数据
构建作者-新闻关系索引，用于 NAN 元路径
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np


class HeterographDataset(Dataset):
    """
    异构图数据集

    从 HeteroData 中提取新闻和作者关系
    用于 NAN (News-Author-News) 元路径
    """

    def __init__(
        self,
        hetero_data,
        mode: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Args:
            hetero_data: PyG HeteroData 对象
            mode: 'train', 'val', 或 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            random_seed: 随机种子
        """
        super(HeterographDataset, self).__init__()
        self.hetero_data = hetero_data
        self.mode = mode

        # 提取新闻特征和标签
        self.news_features = hetero_data['news'].x  # (num_news, embedding_dim)
        self.news_labels = hetero_data['news'].y  # (num_news,)

        # 构建作者-新闻索引
        self.author_to_news = self._build_author_news_index()

        # 为每篇新闻找到其作者的其他新闻
        self.news_to_author_news = self._build_news_to_author_news_mapping()

        # 划分训练/验证/测试集
        self.train_idx, self.val_idx, self.test_idx = self._split_data(
            train_ratio, val_ratio, random_seed
        )

        # 根据模式选择索引
        if mode == 'train':
            self.indices = self.train_idx
        elif mode == 'val':
            self.indices = self.val_idx
        elif mode == 'test':
            self.indices = self.test_idx
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _build_author_news_index(self) -> Dict[int, List[int]]:
        """
        构建作者ID -> 新闻列表的映射

        Returns:
            author_to_news: {author_idx: [news_idx1, news_idx2, ...]}
        """
        edge_index = self.hetero_data['author', 'news'].edge_index
        # edge_index[0]: author indices
        # edge_index[1]: news indices

        author_to_news = {}
        for author_idx, news_idx in zip(edge_index[0], edge_index[1]):
            author_idx = author_idx.item()
            news_idx = news_idx.item()

            if author_idx not in author_to_news:
                author_to_news[author_idx] = []
            author_to_news[author_idx].append(news_idx)

        return author_to_news

    def _build_news_to_author_news_mapping(self) -> Dict[int, torch.Tensor]:
        """
        为每篇新闻，找到其作者的所有新闻（包括自己）

        Returns:
            news_to_author_news: {news_idx: tensor([news_idx1, news_idx2, ...])}
        """
        edge_index = self.hetero_data['author', 'news'].edge_index

        # 首先构建 news -> author 的映射
        news_to_author = {}
        for author_idx, news_idx in zip(edge_index[0], edge_index[1]):
            news_idx = news_idx.item()
            author_idx = author_idx.item()
            news_to_author[news_idx] = author_idx

        # 然后为每篇新闻找到其作者的所有新闻
        news_to_author_news = {}
        for news_idx in range(len(self.news_features)):
            author_idx = news_to_author[news_idx]
            # 该作者的所有新闻
            author_news_list = self.author_to_news[author_idx]
            # 转换为 tensor
            news_to_author_news[news_idx] = torch.tensor(
                author_news_list,
                dtype=torch.long
            )

        return news_to_author_news

    def _split_data(
        self,
        train_ratio: float,
        val_ratio: float,
        random_seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        划分训练/验证/测试集

        Returns:
            train_idx, val_idx, test_idx: 索引数组
        """
        num_samples = len(self.news_features)
        indices = np.arange(num_samples)

        # 设置随机种子
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_size = int(num_samples * train_ratio)
        val_size = int(num_samples * val_ratio)

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        return train_idx, val_idx, test_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        获取一个样本

        Returns:
            news_embedding: (embedding_dim,) 新闻embedding
            author_news_embeddings: (num_author_news, embedding_dim)
                该作者的所有新闻embedding
            label: 标签（标量）
            news_idx: 新闻索引（用于调试）
        """
        # 获取真实的新闻索引
        real_idx = self.indices[idx]

        # 新闻特征
        news_embedding = self.news_features[real_idx]

        # 该作者的所有新闻
        author_news_indices = self.news_to_author_news[real_idx]
        author_news_embeddings = self.news_features[author_news_indices]

        # 标签
        label = self.news_labels[real_idx]

        return news_embedding, author_news_embeddings, label, real_idx


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    自定义 collate 函数，用于批处理
    由于每篇新闻的作者新闻数量不同，需要特殊处理
    """
    news_embeddings = []
    author_news_embeddings_list = []
    labels = []
    news_indices = []

    for news_emb, author_news_emb, label, news_idx in batch:
        news_embeddings.append(news_emb)
        author_news_embeddings_list.append(author_news_emb)
        labels.append(label)
        news_indices.append(news_idx)

    # 堆叠新闻特征和标签
    news_embeddings_batch = torch.stack(news_embeddings, dim=0)
    labels_batch = torch.stack(labels, dim=0)

    # 作者新闻特征保持列表形式（因为长度不同）
    # 在模型内部会逐个处理
    news_indices_batch = torch.tensor(news_indices, dtype=torch.long)

    return news_embeddings_batch, author_news_embeddings_list, labels_batch, news_indices_batch


def create_dataloaders(
    hetero_data,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试 DataLoader

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = HeterographDataset(
        hetero_data,
        mode='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    val_dataset = HeterographDataset(
        hetero_data,
        mode='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    test_dataset = HeterographDataset(
        hetero_data,
        mode='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载器
    print("=" * 60)
    print("Testing DataLoader")
    print("=" * 60)

    # 加载数据
    hetero_data = torch.load('data/metapath_subgraphs/liar_test_news_author_news.pt')

    print(f"\nData info:")
    print(f"  News nodes: {hetero_data['news'].x.shape}")
    print(f"  Labels: {hetero_data['news'].y.shape}")
    print(f"  Unique labels: {torch.unique(hetero_data['news'].y).tolist()}")
    print(f"  Label distribution:")
    for label in torch.unique(hetero_data['news'].y):
        count = (hetero_data['news'].y == label).sum().item()
        print(f"    Label {label}: {count} samples")

    # 创建 DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        hetero_data,
        batch_size=4,
        train_ratio=0.8,
        val_ratio=0.1,
        num_workers=0
    )

    # 测试迭代
    print(f"\n--- Testing train loader ---")
    for batch_idx, (news_emb, author_news_emb_list, labels, news_idx) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  news_embeddings: {news_emb.shape}")
        print(f"  labels: {labels.shape}")
        print(f"  news_indices: {news_idx}")

        # 检查作者新闻的数量
        for i in range(len(author_news_emb_list)):
            print(f"  Sample {i}: author_news shape = {author_news_emb_list[i].shape}")

        if batch_idx >= 2:
            break

    print("\n[PASS] DataLoader test passed!")
