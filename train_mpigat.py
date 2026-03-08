"""
MPI-GAT 训练脚本

训练流程：
1. 加载数据
2. 创建模型
3. 训练/验证循环
4. 保存最佳模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.MPI_GAT.mpi_gat_simple import MPIGAT_Simple
from model.MPI_GAT.dataset import create_dataloaders


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个 epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (news_emb, author_news_emb_list, labels, news_idx) in enumerate(pbar):
        # 移动到设备
        news_emb = news_emb.to(device)
        labels = labels.to(device)

        # 前向传播
        logits, debug_info = model(news_emb, author_news_emb_list)

        # 计算损失
        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
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
    """
    验证
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (news_emb, author_news_emb_list, labels, news_idx) in enumerate(tqdm(val_loader, desc="Validating")):
            # 移动到设备
            news_emb = news_emb.to(device)
            labels = labels.to(device)

            # 前向传播
            logits, debug_info = model(news_emb, author_news_emb_list)

            # 计算损失
            loss = criterion(logits, labels)

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 保存预测结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy, all_predictions, all_labels


def compute_metrics(predictions, labels, num_classes=6):
    """
    计算评估指标
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

    # 转换为 numpy 数组
    predictions = np.array(predictions)
    labels = np.array(labels)

    # 整体准确率
    accuracy = (predictions == labels).mean() * 100

    # 每个类别的 F1, Precision, Recall
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_weighted = f1_score(labels, predictions, average='weighted')

    precision_macro = precision_score(labels, predictions, average='macro')
    recall_macro = recall_score(labels, predictions, average='macro')

    # 每个类别的指标
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
    data_path,
    output_dir='./checkpoints',
    embedding_dim=1024,
    hidden_dims=[512, 256],
    num_classes=6,
    nan_k=3,
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-5,
    patience=10,
    device='cuda'
):
    """
    训练主函数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    hetero_data = torch.load(data_path)

    train_loader, val_loader, test_loader = create_dataloaders(
        hetero_data,
        batch_size=batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        num_workers=0,
        random_seed=42
    )

    # 创建模型
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)
    model = MPIGAT_Simple(
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        nan_k=nan_k,
        dropout_rate=0.5
    ).to(device)

    print(f"\nModel created!")
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

            # 保存模型
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

    # 加载最佳模型
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # 在测试集上评估
    test_loss, test_acc, predictions, labels = validate(model, test_loader, criterion, device)
    metrics = compute_metrics(predictions, labels, num_classes)

    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (micro): {metrics['f1_micro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"  F1 per class: {metrics['f1_per_class']}")

    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_metrics': metrics
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
        'data_path': 'data/metapath_subgraphs/liar_test_news_author_news.pt',
        'output_dir': './checkpoints/mpigat_nan',
        'embedding_dim': 1024,
        'hidden_dims': [512, 256],
        'num_classes': 6,
        'nan_k': 3,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 开始训练
    model, history = train(**config)
