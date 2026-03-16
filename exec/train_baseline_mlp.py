"""
训练 Baseline_MLP (阶段1)

只使用新闻文本，建立弱基线

数据要求:
- data/liar_embedded/train_embedded.json
- data/liar_embedded/valid_embedded.json
- data/liar_embedded/test_embedded.json

输出:
- checkpoints/baseline_mlp/<timestamp>/  每次运行独立目录
  - best_model.pth
  - training_history.pth
  - results_summary.json

使用方式:
  python train_baseline_mlp.py                    # 默认参数训练
  python train_baseline_mlp.py --epochs 100       # 指定轮数
  python train_baseline_mlp.py --runs 5           # 连续跑5次
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
import json
import argparse
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Baseline.mlp_models import Baseline_MLP

# 标签名称
LABEL_NAMES = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']


# =============================================================================
# 数据集定义
# =============================================================================

class LiarDataset(Dataset):
    """
    Liar 数据集加载器
    """

    def __init__(self, data_path):
        print(f"Loading data from: {data_path}")

        self.label_map = {
            'true': 0, 'mostly-true': 1, 'half-true': 2,
            'barely-true': 3, 'false': 4, 'pants-fire': 5
        }

        if data_path.endswith('.json'):
            self._load_json(data_path)
        elif data_path.endswith('.pt'):
            self._load_pt(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        print(f"Loaded {len(self.embeddings)} samples")
        print(f"  Embedding shape: {self.embeddings.shape}")
        print(f"  Label distribution: {torch.bincount(self.labels)}")

    def _load_json(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.embeddings = []
        self.labels = []
        for item in raw_data:
            self.embeddings.append(item["embedding"])
            label = item.get("label", item.get("label_id", 2))
            if isinstance(label, str):
                label = self.label_map.get(label.lower(), 2)
            self.labels.append(label)

        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _load_pt(self, data_path):
        self.data = torch.load(data_path)
        self.news_ids = list(self.data.keys())

        self.embeddings = []
        self.labels = []
        for news_id in self.news_ids:
            item = self.data[news_id]
            self.embeddings.append(item["embedding"])
            self.labels.append(item["label"])

        self.embeddings = torch.stack(self.embeddings)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {"embedding": self.embeddings[idx], "label": self.labels[idx]}


# =============================================================================
# 训练相关函数
# =============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)

        logits = model(embeddings)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100. * correct / total:.2f}%'})

    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
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

    return total_loss / len(val_loader), 100. * correct / total, all_predictions, all_labels


def compute_metrics(predictions, labels, num_classes=6):
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


# =============================================================================
# 结果展示函数
# =============================================================================

def print_results(test_metrics, best_val_loss, best_val_acc, timestamp):
    """打印测试结果"""

    print("\n" + "=" * 80)
    print(f"TEST RESULTS [{timestamp}]")
    print("=" * 80)

    print("\n1. OVERALL METRICS")
    print("-" * 80)
    print(f"Accuracy:          {test_metrics['accuracy']:.2f}%")
    print(f"F1 (Macro):        {test_metrics['f1_macro']*100:.2f}%")
    print(f"F1 (Micro):        {test_metrics['f1_micro']*100:.2f}%")
    print(f"F1 (Weighted):     {test_metrics['f1_weighted']*100:.2f}%")
    print(f"Precision (Macro): {test_metrics['precision_macro']*100:.2f}%")
    print(f"Recall (Macro):    {test_metrics['recall_macro']*100:.2f}%")

    print("\n2. PER-CLASS F1 SCORES")
    print("-" * 80)
    print(f"{'Class':<15} {'F1 Score':>10}")
    print("-" * 80)
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:<15} {test_metrics['f1_per_class'][i]*100:>10.2f}%")

    print("\n3. FOR PAPER (LaTeX Format)")
    print("-" * 80)

    print("\n% Table: Overall Performance")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Performance of Baseline MLP}")
    print("\\begin{tabular}{lc}")
    print("\\hline")
    print("Metric & Value ($\\%$) \\\\")
    print("\\hline")
    print(f"Accuracy & {test_metrics['accuracy']:.2f} \\\\")
    print(f"F1 (Macro) & {test_metrics['f1_macro']*100:.2f} \\\\")
    print(f"F1 (Weighted) & {test_metrics['f1_weighted']*100:.2f} \\\\")
    print(f"Precision (Macro) & {test_metrics['precision_macro']*100:.2f} \\\\")
    print(f"Recall (Macro) & {test_metrics['recall_macro']*100:.2f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n" + "=" * 80)
    print("SUMMARY FOR ABLATION STUDY")
    print("=" * 80)
    print(f"Stage 1 (Baseline_MLP): {test_metrics['accuracy']:.2f}% accuracy")
    print("Input: News text only")
    print("=" * 80)


# =============================================================================
# 训练主函数
# =============================================================================

def train(
    train_path,
    val_path,
    test_path,
    base_output_dir='./checkpoints/baseline_mlp',
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
    训练主函数，返回时间戳和输出目录
    """
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print(f"\nRun timestamp: {timestamp}")
    print(f"Output directory: {output_dir}")

    # 加载数据
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    train_dataset = LiarDataset(train_path)
    val_dataset = LiarDataset(val_path)
    test_dataset = LiarDataset(test_path)

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练循环
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    best_val_acc = 0
    epochs_no_improve = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc, predictions, labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        metrics = compute_metrics(predictions, labels, num_classes)

        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"F1 (macro): {metrics['f1_macro']:.4f}")

        scheduler.step(val_loss)

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
    test_metrics = compute_metrics(predictions, labels, num_classes)

    # 保存训练历史
    history = {
        'timestamp': timestamp,
        'random_seed': random_seed,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics
    }
    torch.save(history, os.path.join(output_dir, 'training_history.pth'))

    # 保存结果摘要
    results_summary = {
        'timestamp': timestamp,
        'model': 'Baseline_MLP',
        'stage': 1,
        'description': 'Text-only baseline',
        'random_seed': random_seed,
        'best_val_loss': float(best_val_loss),
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1_macro': float(test_metrics['f1_macro'] * 100),
        'test_f1_weighted': float(test_metrics['f1_weighted'] * 100),
        'test_precision_macro': float(test_metrics['precision_macro'] * 100),
        'test_recall_macro': float(test_metrics['recall_macro'] * 100),
        'per_class_f1': (test_metrics['f1_per_class'] * 100).tolist()
    }

    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    # 打印结果
    print_results(test_metrics, best_val_loss, best_val_acc, timestamp)

    print(f"\n[OK] Results saved to: {output_dir}")

    return timestamp, output_dir, test_metrics


def compute_average_results(base_output_dir):
    """计算多次运行的平均结果"""

    print("\n" + "=" * 80)
    print("COMPUTING AVERAGE RESULTS")
    print("=" * 80)

    # 查找所有运行目录
    run_dirs = []
    for item in os.listdir(base_output_dir):
        item_path = os.path.join(base_output_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'results_summary.json')):
            run_dirs.append(item_path)

    if len(run_dirs) == 0:
        print("No previous runs found.")
        return None

    run_dirs.sort()
    print(f"Found {len(run_dirs)} runs")

    # 收集所有结果
    all_results = []
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, 'results_summary.json'), 'r') as f:
            result = json.load(f)
            all_results.append(result)

    # 计算平均值和标准差
    metrics_keys = ['test_accuracy', 'test_f1_macro', 'test_f1_weighted',
                    'test_precision_macro', 'test_recall_macro']

    avg_results = {'num_runs': len(all_results)}
    for key in metrics_keys:
        values = [r[key] for r in all_results]
        avg_results[f'{key}_mean'] = np.mean(values)
        avg_results[f'{key}_std'] = np.std(values)

    # Per-class F1
    per_class_f1 = np.array([r['per_class_f1'] for r in all_results])
    avg_results['per_class_f1_mean'] = np.mean(per_class_f1, axis=0).tolist()
    avg_results['per_class_f1_std'] = np.std(per_class_f1, axis=0).tolist()

    # 保存平均结果
    avg_path = os.path.join(base_output_dir, 'average_results.json')
    with open(avg_path, 'w') as f:
        json.dump(avg_results, f, indent=2)

    # 打印平均结果
    print("\n" + "-" * 80)
    print(f"Average over {len(all_results)} runs:")
    print("-" * 80)
    print(f"Accuracy:          {avg_results['test_accuracy_mean']:.2f} +/- {avg_results['test_accuracy_std']:.2f}%")
    print(f"F1 (Macro):        {avg_results['test_f1_macro_mean']:.2f} +/- {avg_results['test_f1_macro_std']:.2f}%")
    print(f"F1 (Weighted):     {avg_results['test_f1_weighted_mean']:.2f} +/- {avg_results['test_f1_weighted_std']:.2f}%")
    print(f"Precision (Macro): {avg_results['test_precision_macro_mean']:.2f} +/- {avg_results['test_precision_macro_std']:.2f}%")
    print(f"Recall (Macro):    {avg_results['test_recall_macro_mean']:.2f} +/- {avg_results['test_recall_macro_std']:.2f}%")

    print("\nPer-Class F1 (mean +/- std):")
    print("-" * 80)
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:<15} {avg_results['per_class_f1_mean'][i]:.2f} +/- {avg_results['per_class_f1_std'][i]:.2f}%")

    print("\n" + "-" * 80)
    print(f"Average results saved to: {avg_path}")

    return avg_results


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Baseline MLP')

    parser.add_argument('--train-path', type=str,
        default='data/liar_embedded/train_embedded.json', help='Training data path')
    parser.add_argument('--val-path', type=str,
        default='data/liar_embedded/valid_embedded.json', help='Validation data path')
    parser.add_argument('--test-path', type=str,
        default='data/liar_embedded/test_embedded.json', help='Test data path')
    parser.add_argument('--output-dir', type=str,
        default='./checkpoints/baseline_mlp', help='Base output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs (for averaging)')
    parser.add_argument('--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

    args = parser.parse_args()

    # 打印配置
    print("=" * 60)
    print("Training Baseline_MLP (Stage 1)")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  train_path: {args.train_path}")
    print(f"  val_path: {args.val_path}")
    print(f"  test_path: {args.test_path}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.lr}")
    print(f"  patience: {args.patience}")
    print(f"  runs: {args.runs}")
    print(f"  device: {args.device}")

    # 检查数据文件
    for path_key in ['train_path', 'val_path', 'test_path']:
        path = getattr(args, path_key)
        if not os.path.exists(path):
            print(f"\n[ERROR] Data file not found: {path}")
            sys.exit(1)

    # 多次运行
    for run_idx in range(args.runs):
        print(f"\n{'='*80}")
        print(f"RUN {run_idx + 1}/{args.runs}")
        print(f"{'='*80}")

        # 使用不同的随机种子
        random_seed = 42 + run_idx

        train(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            base_output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            random_seed=random_seed,
            device=args.device
        )

    # 计算平均结果
    if args.runs > 1:
        compute_average_results(args.output_dir)

    print("\n" + "=" * 80)
    print("ALL RUNS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
