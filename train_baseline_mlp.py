"""
训练 Baseline_MLP (阶段1)

只使用新闻文本，建立弱基线

数据要求:
- data/liar_clean_embeddings.pt

输出:
- checkpoints/baseline_mlp/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Baseline.mlp_models import Baseline_MLP


class LiarDataset(Dataset):
    """
    Liar 数据集加载器

    支持的数据格式:
    1. JSON格式 (liar_embedded):
        [
            {
                "statement": "...",
                "embedding": [1024维],
                "label": "false"
            },
            ...
        ]

    2. PT格式 (torch saved):
        {
            "news_id": {
                "embedding": [1024维],
                "label": int
            }
        }
    """

    def __init__(self, data_path):
        print(f"Loading data from: {data_path}")

        # 标签映射
        self.label_map = {
            'true': 0,
            'mostly-true': 1,
            'half-true': 2,
            'barely-true': 3,
            'false': 4,
            'pants-fire': 5
        }

        # 根据文件扩展名判断格式
        if data_path.endswith('.json'):
            self._load_json(data_path)
        elif data_path.endswith('.pt'):
            self._load_pt(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        print(f"Loaded {len(self.embeddings)} samples")
        print(f"  Embedding shape: {self.embeddings.shape}")
        print(f"  Label shape: {self.labels.shape}")
        print(f"  Label distribution: {torch.bincount(self.labels)}")

    def _load_json(self, data_path):
        """加载JSON格式数据"""
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 提取数据
        self.embeddings = []
        self.labels = []
        self.statements = []

        for item in raw_data:
            self.embeddings.append(item["embedding"])
            # 标签可能是字符串或数字
            label = item.get("label", item.get("label_id", 2))
            if isinstance(label, str):
                label = self.label_map.get(label.lower(), 2)
            self.labels.append(label)
            self.statements.append(item.get("statement", ""))

        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _load_pt(self, data_path):
        """加载PT格式数据"""
        self.data = torch.load(data_path)
        self.news_ids = list(self.data.keys())

        for news_id in self.news_ids:
            item = self.data[news_id]
            self.embeddings.append(item["embedding"])
            self.labels.append(item["label"])

        self.embeddings = torch.stack(self.embeddings)
        self.labels = torch.tensor(self.labels)
        self.statements = [""] * len(self.news_ids)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx]
        }


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)

        # 前向传播
        logits = model(embeddings)

        # 计算损失
        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)

            logits = model(embeddings)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy, all_predictions, all_labels


def compute_metrics(predictions, labels, num_classes=6):
    """计算评估指标"""
    from sklearn.metrics import f1_score, precision_score, recall_score

    predictions = np.array(predictions)
    labels = np.array(labels)

    accuracy = (predictions == labels).mean() * 100

    f1_macro = f1_score(labels, predictions, average='macro')
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_weighted = f1_score(labels, predictions, average='weighted')

    precision_macro = precision_score(labels, predictions, average='macro')
    recall_macro = recall_score(labels, predictions, average='macro')

    f1_per_class = f1_score(labels, predictions, average=None)

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_per_class': f1_per_class
    }

    return metrics


def train(
    train_path,
    val_path=None,
    test_path=None,
    output_dir='./checkpoints/baseline_mlp',
    embedding_dim=1024,
    hidden_dims=[512, 256],
    num_classes=6,
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-5,
    patience=10,
    random_seed=42,
    device='cuda'
):
    """
    训练主函数
    """
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    train_dataset = LiarDataset(train_path)

    # 如果没有提供验证集和测试集路径，则从训练集划分
    if val_path is None or test_path is None:
        print("\nNo validation/test paths provided, splitting from train set...")
        total_size = len(train_dataset)
        val_size = int(total_size * 0.1)
        test_size = int(total_size * 0.1)
        train_size = total_size - val_size - test_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
    else:
        val_dataset = LiarDataset(val_path)
        test_dataset = LiarDataset(test_path)

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 创建模型
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)

    model = Baseline_MLP(
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rate=0.5
    ).to(device)

    print(f"\nModel: Baseline_MLP")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练循环
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    best_val_acc = 0
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        val_loss, val_acc, predictions, labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 计算详细指标
        metrics = compute_metrics(predictions, labels, num_classes)

        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"F1 (macro): {metrics['f1_macro']:.4f} | F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"F1 per class: {metrics['f1_per_class']}")

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'metrics': metrics
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            print(f"[BEST] Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # 测试最佳模型
    print("\n" + "=" * 60)
    print("Testing best model...")
    print("=" * 60)

    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, predictions, labels = validate(model, test_loader, criterion, device)
    metrics = compute_metrics(predictions, labels, num_classes)

    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (micro): {metrics['f1_micro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"  F1 per class: {metrics['f1_per_class']}")

    # 保存训练历史（包含测试集预测结果）
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_metrics': metrics,
        'test_predictions': predictions,  # 添加预测结果
        'test_labels': labels  # 添加真实标签
    }
    torch.save(history, os.path.join(output_dir, 'training_history.pth'))

    print(f"\nTraining completed!")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Test Acc: {test_acc:.2f}%")
    print(f"  Checkpoints saved to: {output_dir}")

    return model, history


if __name__ == "__main__":
    # 配置
    config = {
        'train_path': 'data/liar_embedded/train_embedded.json',
        'val_path': 'data/liar_embedded/valid_embedded.json',
        'test_path': 'data/liar_embedded/test_embedded.json',
        'output_dir': './checkpoints/baseline_mlp',
        'embedding_dim': 1024,
        'hidden_dims': [512, 256],
        'num_classes': 6,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 10,
        'random_seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 60)
    print("Training Baseline_MLP (Stage 1)")
    print("=" * 60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 检查数据文件
    for path_key in ['train_path', 'val_path', 'test_path']:
        if not os.path.exists(config[path_key]):
            print(f"\n[ERROR] Data file not found: {config[path_key]}")
            sys.exit(1)

    # 开始训练
    model, history = train(**config)
