"""
阶段三：GNN 模型训练脚本

支持 GCN/GAT 模型训练
多次运行取平均

使用方式:
  python train_gnn.py --model gcn --runs 5
  python train_gnn.py --model gat --runs 5
  python train_gnn.py --model avgpool --runs 5  # 基线对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import sys
import json
import argparse
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.GNN.dataset import create_dataloaders, LABEL_NAMES
from model.GNN.gnn_models import CommentGCN, CommentGAT, AvgPoolBaseline


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        news_feats = batch['news_feats'].to(device)
        comment_feats_list = batch['comment_feats_list']
        labels = batch['labels'].to(device)

        # 前向传播
        logits, debug_info = model(news_feats, comment_feats_list)

        # 计算损失
        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return total_loss / len(train_loader), 100. * correct / total


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
            news_feats = batch['news_feats'].to(device)
            comment_feats_list = batch['comment_feats_list']
            labels = batch['labels'].to(device)

            logits, debug_info = model(news_feats, comment_feats_list)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(val_loader), 100. * correct / total, all_predictions, all_labels


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

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_per_class': f1_per_class
    }


def train(
    model_name: str,
    data_dir: str,
    output_dir: str,
    embedding_dim: int = 1024,
    hidden_dims: list = [256, 128],
    num_classes: int = 6,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    random_seed: int = 42,
    device: str = 'cuda'
):
    """训练主函数"""
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_base_dir=data_dir,
        batch_size=batch_size,
        max_comments=50,
        embedding_dim=embedding_dim
    )

    # 创建模型
    print("\n" + "=" * 60)
    print(f"Creating {model_name.upper()} model...")
    print("=" * 60)

    if model_name == 'gcn':
        model = CommentGCN(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes
        )
    elif model_name == 'gat':
        model = CommentGAT(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            heads=1
        )
    elif model_name == 'avgpool':
        model = AvgPoolBaseline(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    print(f"\nModel: {model.__class__.__name__}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    # 使用标签平滑减少过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0
    epochs_no_improve = 0

    # 训练历史记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, predictions, labels = validate(model, val_loader, criterion, device)
        val_metrics = compute_metrics(predictions, labels, num_classes)

        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"F1 (macro): {val_metrics['f1_macro']:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_metrics': val_metrics
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"[BEST] Saved model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1

        # 保存训练历史（每个epoch后）
        training_history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_loss': float(best_val_loss),
            'best_val_acc': float(best_val_acc),
            'best_epoch': best_epoch
        }
        torch.save(training_history, os.path.join(output_dir, 'training_history.pth'))

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # 测试
    print("\n" + "=" * 60)
    print("Testing best model...")
    print("=" * 60)

    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, predictions, labels = validate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(predictions, labels, num_classes)

    # 保存结果
    results = {
        'model': model_name,
        'random_seed': random_seed,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'best_val_acc': float(best_val_acc),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1_macro': float(test_metrics['f1_macro'] * 100),
        'test_f1_weighted': float(test_metrics['f1_weighted'] * 100),
        'test_precision_macro': float(test_metrics['precision_macro'] * 100),
        'test_recall_macro': float(test_metrics['recall_macro'] * 100),
        'per_class_f1': (test_metrics['f1_per_class'] * 100).tolist()
    }

    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 打印结果
    print_results(test_metrics, model_name)

    return test_metrics


def print_results(metrics, model_name):
    """打印测试结果"""
    print("\n" + "=" * 80)
    print(f"TEST RESULTS - {model_name.upper()}")
    print("=" * 80)

    print("\n1. OVERALL METRICS")
    print("-" * 80)
    print(f"Accuracy:          {metrics['accuracy']:.2f}%")
    print(f"F1 (Macro):        {metrics['f1_macro']*100:.2f}%")
    print(f"F1 (Weighted):     {metrics['f1_weighted']*100:.2f}%")
    print(f"Precision (Macro): {metrics['precision_macro']*100:.2f}%")
    print(f"Recall (Macro):    {metrics['recall_macro']*100:.2f}%")

    print("\n2. PER-CLASS F1 SCORES")
    print("-" * 80)
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:<15} {metrics['f1_per_class'][i]*100:>10.2f}%")


def compute_average_results(output_dir, num_runs):
    """计算多次运行的平均结果"""
    all_results = []

    for run_idx in range(num_runs):
        result_file = os.path.join(output_dir, f'run_{run_idx}', 'results_summary.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                all_results.append(json.load(f))

    if not all_results:
        print("No results found!")
        return None

    # 计算平均值和标准差
    avg_results = {
        'num_runs': len(all_results),
        'test_accuracy_mean': np.mean([r['test_accuracy'] for r in all_results]),
        'test_accuracy_std': np.std([r['test_accuracy'] for r in all_results]),
        'test_f1_macro_mean': np.mean([r['test_f1_macro'] for r in all_results]),
        'test_f1_macro_std': np.std([r['test_f1_macro'] for r in all_results]),
    }

    # 保存平均结果
    with open(os.path.join(output_dir, 'average_results.json'), 'w') as f:
        json.dump(avg_results, f, indent=2)

    # 打印
    print("\n" + "=" * 80)
    print(f"AVERAGE RESULTS ({len(all_results)} runs)")
    print("=" * 80)
    print(f"Accuracy: {avg_results['test_accuracy_mean']:.2f} ± {avg_results['test_accuracy_std']:.2f}%")
    print(f"F1 Macro: {avg_results['test_f1_macro_mean']:.2f} ± {avg_results['test_f1_macro_std']:.2f}%")

    return avg_results


def main():
    parser = argparse.ArgumentParser(description='Train GNN models for fake news detection')

    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'avgpool'],
                        help='Model type: gcn, gat, or avgpool')
    parser.add_argument('--data-dir', type=str, default='data/comment_networks_embeddings',
                        help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/gnn',
                        help='Output directory')
    parser.add_argument('--embedding-dim', type=int, default=1024)
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--num-classes', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("=" * 60)
    print(f"Stage 3: GNN Training ({args.model.upper()})")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Runs: {args.runs}")
    print(f"Device: {args.device}")

    # 多次运行
    for run_idx in range(args.runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{args.runs}")
        print(f"{'='*60}")

        run_output_dir = os.path.join(args.output_dir, args.model, f'run_{run_idx}')

        train(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=run_output_dir,
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
            random_seed=42 + run_idx,
            device=args.device
        )

    # 计算平均结果
    compute_average_results(os.path.join(args.output_dir, args.model), args.runs)

    print("\n" + "=" * 60)
    print("ALL RUNS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
