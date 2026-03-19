"""
GNN 数据加载器

读取 data/comment_networks_embeddings/ 下的 JSON 文件
构建新闻-评论二部图数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
import os
import glob
from tqdm import tqdm


# 标签映射
LABEL_MAP = {
    'true': 0,
    'mostly-true': 1,
    'half-true': 2,
    'barely-true': 3,
    'false': 4,
    'pants-fire': 5
}

LABEL_NAMES = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']


class CommentGraphDataset(Dataset):
    """
    评论图数据集

    每个样本是一个小图：一条新闻 + 它的评论
    """

    def __init__(
        self,
        data_dir: str,
        max_comments: int = 50,
        embedding_dim: int = 1024,
        is_training: bool = False,
        comment_dropout_range: tuple = (0.5, 1.0),
        shuffle_comments: bool = True
    ):
        """
        Args:
            data_dir: 数据目录路径 (train/valid/test)
            max_comments: 最大评论数量限制
            embedding_dim: 嵌入维度
            is_training: 是否为训练模式（启用数据增强）
            comment_dropout_range: 训练时随机保留评论的比例范围 (min, max)
            shuffle_comments: 是否打乱评论顺序
        """
        self.data_dir = data_dir
        self.max_comments = max_comments
        self.embedding_dim = embedding_dim
        self.is_training = is_training
        self.comment_dropout_range = comment_dropout_range
        self.shuffle_comments = shuffle_comments

        # 加载数据
        self.samples = self._load_data()

        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        if self.is_training:
            print(f"  Training mode: comment_dropout={comment_dropout_range}, shuffle={shuffle_comments}")

        # 统计标签分布
        label_counts = {}
        for sample in self.samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"Label distribution: {label_counts}")

    def _load_data(self) -> List[Dict]:
        """加载所有 JSON 文件"""
        samples = []

        json_files = glob.glob(os.path.join(self.data_dir, '*.json'))
        print(f"Found {len(json_files)} JSON files in {self.data_dir}")

        for json_file in tqdm(json_files, desc="Loading data"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取新闻特征
                news_embedding = data.get('news_embedding')
                if news_embedding is None:
                    # 如果没有 news_embedding，跳过
                    continue

                # 提取标签
                label_str = data.get('news_label', data.get('label', 'half-true'))
                if isinstance(label_str, str):
                    label = LABEL_MAP.get(label_str.lower(), 2)  # 默认 half-true
                else:
                    label = label_str

                # 提取评论特征
                comments = data.get('comments', [])
                comment_embeddings = []
                for comment in comments:
                    emb = comment.get('embedding')
                    if emb is not None:
                        comment_embeddings.append(emb)

                # 限制评论数量
                if len(comment_embeddings) > self.max_comments:
                    comment_embeddings = comment_embeddings[:self.max_comments]

                samples.append({
                    'news_embedding': np.array(news_embedding, dtype=np.float32),
                    'comment_embeddings': comment_embeddings,  # List of arrays
                    'label': label,
                    'news_id': data.get('news_id', json_file)
                })

            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # 新闻特征
        news_feat = torch.tensor(sample['news_embedding'], dtype=torch.float32)

        # 评论特征（可能有 0 条评论）
        comment_feats = sample['comment_embeddings']

        # 数据增强：训练时应用评论dropout和打乱
        if self.is_training and len(comment_feats) > 0:
            # 1. 随机保留一定比例的评论（评论dropout）
            keep_ratio = np.random.uniform(self.comment_dropout_range[0], self.comment_dropout_range[1])
            num_keep = max(1, int(len(comment_feats) * keep_ratio))

            # 随机采样评论
            indices = np.random.choice(len(comment_feats), size=num_keep, replace=False)
            comment_feats = [comment_feats[i] for i in indices]

            # 2. 打乱评论顺序
            if self.shuffle_comments:
                np.random.shuffle(comment_feats)

        if len(comment_feats) > 0:
            comment_feats = torch.tensor(np.array(comment_feats), dtype=torch.float32)

            # 3. 添加小噪声（仅训练时）
            if self.is_training:
                noise = torch.randn_like(comment_feats) * 0.01
                comment_feats = comment_feats + noise
        else:
            # 如果没有评论，创建一个全零的占位符
            comment_feats = torch.zeros((1, self.embedding_dim), dtype=torch.float32)

        # 评论数量（用于某些模型）
        num_comments = torch.tensor(len(sample['comment_embeddings']), dtype=torch.long)

        # 标签
        label = torch.tensor(sample['label'], dtype=torch.long)

        return {
            'news_feat': news_feat,           # (embedding_dim,)
            'comment_feats': comment_feats,   # (num_comments, embedding_dim)
            'num_comments': num_comments,     # scalar
            'label': label,                   # scalar
            'news_id': sample['news_id']
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义批处理函数

    由于每个样本的评论数量不同，需要特殊处理
    """
    news_feats = []
    comment_feats_list = []
    num_comments_list = []
    labels = []
    news_ids = []

    for item in batch:
        news_feats.append(item['news_feat'])
        comment_feats_list.append(item['comment_feats'])
        num_comments_list.append(item['num_comments'])
        labels.append(item['label'])
        news_ids.append(item['news_id'])

    # 堆叠新闻特征和标签
    news_feats = torch.stack(news_feats, dim=0)  # (batch_size, embedding_dim)
    labels = torch.stack(labels, dim=0)           # (batch_size,)
    num_comments = torch.stack(num_comments_list, dim=0)  # (batch_size,)

    return {
        'news_feats': news_feats,
        'comment_feats_list': comment_feats_list,  # List of tensors
        'num_comments': num_comments,
        'labels': labels,
        'news_ids': news_ids
    }


def create_dataloaders(
    data_base_dir: str = 'data/comment_networks_embeddings',
    batch_size: int = 32,
    max_comments: int = 50,
    embedding_dim: int = 1024,
    num_workers: int = 0,
    comment_dropout_range: tuple = (0.5, 1.0),
    shuffle_comments: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试 DataLoader

    Args:
        data_base_dir: 数据基础目录
        batch_size: 批大小
        max_comments: 最大评论数量
        embedding_dim: 嵌入维度
        num_workers: 数据加载线程数
        comment_dropout_range: 训练时评论dropout比例范围
        shuffle_comments: 是否打乱评论顺序

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dir = os.path.join(data_base_dir, 'train')
    val_dir = os.path.join(data_base_dir, 'valid')
    test_dir = os.path.join(data_base_dir, 'test')

    train_dataset = CommentGraphDataset(
        train_dir,
        max_comments=max_comments,
        embedding_dim=embedding_dim,
        is_training=True,
        comment_dropout_range=comment_dropout_range,
        shuffle_comments=shuffle_comments
    )
    val_dataset = CommentGraphDataset(
        val_dir,
        max_comments=max_comments,
        embedding_dim=embedding_dim,
        is_training=False
    )
    test_dataset = CommentGraphDataset(
        test_dir,
        max_comments=max_comments,
        embedding_dim=embedding_dim,
        is_training=False
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

    print(f"\nDataset summary:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载器
    print("=" * 60)
    print("Testing CommentGraphDataset")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_base_dir='data/comment_networks_embeddings',
        batch_size=4,
        max_comments=50,
        embedding_dim=1024
    )

    # 测试迭代
    print("\n--- Testing train loader ---")
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  news_feats: {batch['news_feats'].shape}")
        print(f"  num_comments: {batch['num_comments']}")
        print(f"  labels: {batch['labels']}")
        print(f"  comment_feats_list length: {len(batch['comment_feats_list'])}")
        for i, cf in enumerate(batch['comment_feats_list']):
            print(f"    Sample {i}: comment_feats shape = {cf.shape}")

        if batch_idx >= 2:
            break

    print("\n[PASS] DataLoader test passed!")
