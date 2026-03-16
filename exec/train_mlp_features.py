"""
训练 Feature_MLP (阶段2 消融实验)

使用新闻文本 + 额外特征 (作者/评论)

融合方式:
- concat: 直接拼接 (默认)
- residual: 残差融合 (保证不比只用新闻差)

数据要求:
- data/comment_networks_embeddings/train/*.json
- data/comment_networks_embeddings/test/*.json

使用方式:
  python train_mlp_features.py --features author --fusion concat
  python train_mlp_features.py --features author --fusion residual
  python train_mlp_features.py --features author,comment --fusion residual --runs 5
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
import glob
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Baseline.mlp_models import Feature_MLP_Concat, Feature_MLP_Residual

# 标签名称
LABEL_NAMES = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']


# =============================================================================
# 数据集定义
# =============================================================================

class LiarFeatureDataset(Dataset):
    """Liar 数据集加载器 (新闻 + 额外特征)"""

    def __init__(self, data_dir, features=['author']):
        print(f"Loading data from: {data_dir}")
        print(f"Extra features: {features}")

        self.features = features
        self.label_map = {
            'true': 0, 'mostly-true': 1, 'half-true': 2,
            'barely-true': 3, 'false': 4, 'pants-fire': 5
        }

        json_files = glob.glob(os.path.join(data_dir, '*.json'))
        print(f"Found {len(json_files)} JSON files")

        self.news_embeddings = []
        self.extra_features = []
        self.labels = []

        # 残差融合需要单独存储 author 和 comment
        self.author_embeddings = []
        self.comment_embeddings = []

        for json_file in tqdm(json_files, desc="Loading"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._process_item(data)

        self.news_embeddings = torch.tensor(self.news_embeddings, dtype=torch.float32)
        self.extra_features = torch.tensor(self.extra_features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # 残差融合用
        if self.author_embeddings:
            self.author_embeddings = torch.tensor(self.author_embeddings, dtype=torch.float32)
        if self.comment_embeddings:
            self.comment_embeddings = torch.tensor(self.comment_embeddings, dtype=torch.float32)

        print(f"Loaded {len(self.labels)} samples")
        print(f"  News embedding shape: {self.news_embeddings.shape}")
        print(f"  Extra features shape: {self.extra_features.shape}")
        print(f"  Label distribution: {torch.bincount(self.labels)}")

    def _process_item(self, data):
        news_emb = data.get('news_embedding')
        if news_emb is None:
            return
        self.news_embeddings.append(news_emb)

        label = data.get('news_label', data.get('label', 'half-true'))
        if isinstance(label, str):
            label = self.label_map.get(label.lower(), 2)
        self.labels.append(label)

        # 单独存储 author 和 comment (残差融合用)
        author_emb = data.get('author_embedding')
        if author_emb:
            self.author_embeddings.append(author_emb)
        else:
            self.author_embeddings.append([0.0] * 1024)

        # 评论平均
        comments = data.get('comments', [])
        if comments:
            comment_embs = [c['embedding'] for c in comments if 'embedding' in c]
            if comment_embs:
                avg_emb = np.mean(comment_embs, axis=0).tolist()
                self.comment_embeddings.append(avg_emb)
            else:
                self.comment_embeddings.append([0.0] * 1024)
        else:
            self.comment_embeddings.append([0.0] * 1024)

        # 拼接特征 (concat 融合用)
        extra_parts = []
        for feat in self.features:
            if feat == 'author':
                extra_parts.append(self.author_embeddings[-1])
            elif feat == 'comment':
                extra_parts.append(self.comment_embeddings[-1])
        self.extra_features.append(np.concatenate(extra_parts))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "news_embedding": self.news_embeddings[idx],
            "extra_features": self.extra_features[idx],
            "label": self.labels[idx]
        }
        # 残差融合用
        if len(self.author_embeddings) > 0:
            item["author_embedding"] = self.author_embeddings[idx]
        if len(self.comment_embeddings) > 0:
            item["comment_embedding"] = self.comment_embeddings[idx]
        return item


# =============================================================================
# 训练相关函数
# =============================================================================

def train_epoch_concat(model, train_loader, optimizer, criterion, device):
    """训练一个 epoch (concat 融合)"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        news_emb = batch["news_embedding"].to(device)
        extra_feat = batch["extra_features"].to(device)
        labels = batch["label"].to(device)

        logits = model(news_emb, extra_feat)
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


def validate_concat(model, val_loader, criterion, device):
    """验证 (concat 融合)"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            news_emb = batch["news_embedding"].to(device)
            extra_feat = batch["extra_features"].to(device)
            labels = batch["label"].to(device)

            logits = model(news_emb, extra_feat)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(val_loader), 100. * correct / total, all_predictions, all_labels


def train_epoch_residual(model, train_loader, optimizer, criterion, device):
    """训练一个 epoch (residual 融合)"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        news_emb = batch["news_embedding"].to(device)
        author_emb = batch["author_embedding"].to(device)
        comment_emb = batch["comment_embedding"].to(device)
        labels = batch["label"].to(device)

        logits = model(news_emb, author_emb, comment_emb)
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


def validate_residual(model, val_loader, criterion, device):
    """验证 (residual 融合)"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            news_emb = batch["news_embedding"].to(device)
            author_emb = batch["author_embedding"].to(device)
            comment_emb = batch["comment_embedding"].to(device)
            labels = batch["label"].to(device)

            logits = model(news_emb, author_emb, comment_emb)
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


def print_results(test_metrics, feature_name, fusion_type):
    print("\n" + "=" * 80)
    print(f"TEST RESULTS - {feature_name} ({fusion_type})")
    print("=" * 80)

    print("\n1. OVERALL METRICS")
    print("-" * 80)
    print(f"Accuracy:          {test_metrics['accuracy']:.2f}%")
    print(f"F1 (Macro):        {test_metrics['f1_macro']*100:.2f}%")
    print(f"F1 (Weighted):     {test_metrics['f1_weighted']*100:.2f}%")

    print("\n2. PER-CLASS F1 SCORES")
    print("-" * 80)
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:<15} {test_metrics['f1_per_class'][i]*100:>10.2f}%")


# =============================================================================
# 训练主函数
# =============================================================================

def train(
    train_dir,
    val_dir,
    test_dir,
    features,
    fusion='concat',
    base_output_dir='./checkpoints/mlp_ablation',
    embedding_dim=1024,
    hidden_dims=[512, 256],
    hidden_dim=256,  # 残差融合用
    num_classes=6,
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-5,
    patience=10,
    random_seed=42,
    device='cuda'
):
    feature_name = "news_" + "_".join(sorted(features))
    
    # 输出目录包含融合方式
    output_dir_name = f"{feature_name}_{fusion}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, output_dir_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print(f"\nRun timestamp: {timestamp}")
    print(f"Output directory: {output_dir}")
    print(f"Fusion type: {fusion}")

    # 加载数据
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    train_dataset = LiarFeatureDataset(train_dir, features=features)
    val_dataset = LiarFeatureDataset(val_dir, features=features)
    test_dataset = LiarFeatureDataset(test_dir, features=features)

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 创建模型
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)

    if fusion == 'concat':
        model = Feature_MLP_Concat(
            embedding_dim=embedding_dim,
            num_extra_features=len(features),
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=0.5
        ).to(device)
        train_epoch_fn = train_epoch_concat
        validate_fn = validate_concat
    else:  # residual
        model = Feature_MLP_Residual(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=0.5
        ).to(device)
        train_epoch_fn = train_epoch_residual
        validate_fn = validate_residual

    print(f"\nModel: {model.__class__.__name__}")
    print(f"  Features: {features}")
    print(f"  Fusion: {fusion}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, predictions, labels = validate_fn(model, val_loader, criterion, device)
        metrics = compute_metrics(predictions, labels, num_classes)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"[BEST] val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # 测试
    print("\n" + "=" * 60)
    print("Testing...")
    print("=" * 60)

    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, predictions, labels = validate_fn(model, test_loader, criterion, device)
    test_metrics = compute_metrics(predictions, labels, num_classes)

    # 保存结果
    results = {
        'timestamp': timestamp,
        'features': features,
        'fusion': fusion,
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1_macro': float(test_metrics['f1_macro'] * 100),
        'test_f1_weighted': float(test_metrics['f1_weighted'] * 100),
        'per_class_f1': (test_metrics['f1_per_class'] * 100).tolist()
    }
    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print_results(test_metrics, feature_name, fusion)
    print(f"\n[OK] Results saved to: {output_dir}")

    return test_metrics


def compute_average_results(feature_dir):
    """计算平均结果"""
    run_dirs = []
    for item in os.listdir(feature_dir):
        item_path = os.path.join(feature_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'results_summary.json')):
            run_dirs.append(item_path)

    if not run_dirs:
        return None

    all_results = []
    for run_dir in sorted(run_dirs):
        with open(os.path.join(run_dir, 'results_summary.json'), 'r') as f:
            all_results.append(json.load(f))

    avg = {
        'num_runs': len(all_results),
        'test_accuracy_mean': np.mean([r['test_accuracy'] for r in all_results]),
        'test_accuracy_std': np.std([r['test_accuracy'] for r in all_results]),
        'test_f1_macro_mean': np.mean([r['test_f1_macro'] for r in all_results]),
        'test_f1_macro_std': np.std([r['test_f1_macro'] for r in all_results]),
    }

    with open(os.path.join(feature_dir, 'average_results.json'), 'w') as f:
        json.dump(avg, f, indent=2)

    print(f"\nAverage over {len(all_results)} runs:")
    print(f"  Accuracy: {avg['test_accuracy_mean']:.2f} ± {avg['test_accuracy_std']:.2f}%")
    print(f"  F1 Macro: {avg['test_f1_macro_mean']:.2f} ± {avg['test_f1_macro_std']:.2f}%")

    return avg


def main():
    parser = argparse.ArgumentParser(description='Train Feature MLP')

    parser.add_argument('--train-dir', type=str,
        default='data/comment_networks_embeddings/train')
    parser.add_argument('--val-dir', type=str,
        default='data/comment_networks_embeddings/valid')
    parser.add_argument('--test-dir', type=str,
        default='data/comment_networks_embeddings/test')
    parser.add_argument('--features', type=str, default='author',
        help='Features: author, comment, or author,comment')
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'residual'],
        help='Fusion type: concat (default) or residual')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/mlp_ablation')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # 解析 features
    features = [f.strip() for f in args.features.split(',')]
    for f in features:
        if f not in ['author', 'comment']:
            print(f"[ERROR] Unknown feature: {f}")
            sys.exit(1)

    print("=" * 60)
    print("Training Feature MLP")
    print("=" * 60)
    print(f"Features: {features}")
    print(f"Fusion: {args.fusion}")
    print(f"Device: {args.device}")

    for run_idx in range(args.runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{args.runs}")
        print(f"{'='*60}")

        train(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            features=features,
            fusion=args.fusion,
            base_output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            random_seed=42 + run_idx,
            device=args.device
        )

    if args.runs > 1:
        feature_dir = os.path.join(args.output_dir, f"news_{'_'.join(sorted(features))}_{args.fusion}")
        compute_average_results(feature_dir)

    print("\n" + "=" * 60)
    print("ALL RUNS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
