"""
分析模型结果，生成论文所需的评估指标和可视化

输出：
1. 详细的评估指标表格
2. 混淆矩阵
3. 每个类别的详细指标
4. LaTeX表格（用于论文）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import json
import os
from pathlib import Path


def load_results(checkpoint_dir):
    """加载训练结果"""
    history_path = os.path.join(checkpoint_dir, 'training_history.pth')
    model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    history = torch.load(history_path)
    checkpoint = torch.load(model_path)

    return history, checkpoint


def compute_detailed_metrics(predictions, labels, num_classes=6):
    """计算详细的评估指标"""
    label_names = ['true', 'mostly-true', 'half-true',
                   'barely-true', 'false', 'pants-fire']

    # 转换为numpy数组
    predictions = np.array(predictions)
    labels = np.array(labels)

    # 基本指标
    accuracy = accuracy_score(labels, predictions) * 100

    # F1分数
    f1_macro = f1_score(labels, predictions, average='macro') * 100
    f1_micro = f1_score(labels, predictions, average='micro') * 100
    f1_weighted = f1_score(labels, predictions, average='weighted') * 100
    f1_per_class = f1_score(labels, predictions, average=None) * 100

    # Precision和Recall
    precision_macro = precision_score(labels, predictions, average='macro') * 100
    recall_macro = recall_score(labels, predictions, average='macro') * 100

    # 每个类别的指标
    precision_per_class = precision_score(labels, predictions, average=None) * 100
    recall_per_class = recall_score(labels, predictions, average=None) * 100

    # 每个类别的准确率
    accuracies_per_class = []
    for i in range(num_classes):
        mask = (labels == i)
        mask_sum = np.sum(mask) if isinstance(mask, np.ndarray) else int(mask.sum())
        if mask_sum > 0:
            correct_preds = predictions[mask] == i
            correct_sum = np.sum(correct_preds) if isinstance(correct_preds, np.ndarray) else int(correct_preds.sum())
            acc = correct_sum / mask_sum * 100
        else:
            acc = 0
        accuracies_per_class.append(acc)

    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'accuracies_per_class': accuracies_per_class,
        'confusion_matrix': cm,
        'label_names': label_names
    }

    return metrics


def print_detailed_results(metrics):
    """打印详细结果"""
    print("\n" + "=" * 80)
    print("DETAILED EVALUATION RESULTS")
    print("=" * 80)

    print("\n1. OVERALL METRICS")
    print("-" * 80)
    print(f"Accuracy:          {metrics['accuracy']:.2f}%")
    print(f"F1 (Macro):        {metrics['f1_macro']:.2f}%")
    print(f"F1 (Micro):        {metrics['f1_micro']:.2f}%")
    print(f"F1 (Weighted):     {metrics['f1_weighted']:.2f}%")
    print(f"Precision (Macro): {metrics['precision_macro']:.2f}%")
    print(f"Recall (Macro):    {metrics['recall_macro']:.2f}%")

    print("\n2. PER-CLASS METRICS")
    print("-" * 80)
    print(f"{'Class':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)

    for i, name in enumerate(metrics['label_names']):
        print(f"{name:<15} "
              f"{metrics['accuracies_per_class'][i]:>10.2f} "
              f"{metrics['precision_per_class'][i]:>10.2f} "
              f"{metrics['recall_per_class'][i]:>10.2f} "
              f"{metrics['f1_per_class'][i]:>10.2f}")

    print("\n3. CONFUSION MATRIX")
    print("-" * 80)
    cm = metrics['confusion_matrix']
    print("Predicted →")
    print("Actual ↓")
    print(cm)

    print("\n" + "=" * 80)


def plot_confusion_matrix(metrics, save_path):
    """绘制混淆矩阵"""
    cm = metrics['confusion_matrix']
    label_names = metrics['label_names']

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Baseline MLP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


def generate_latex_tables(metrics, output_dir):
    """生成LaTeX表格（用于论文）"""

    # Table 1: Overall Results
    latex_table1 = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Overall Performance of Baseline MLP}}
\\label{{tab:baseline_overall}}
\\begin{{tabular}}{{lcc}}
\\hline
Metric & Score ($\\%$) \\\\
\\hline
Accuracy & {metrics['accuracy']:.2f} \\\\
F1 (Macro) & {metrics['f1_macro']:.2f} \\\\
F1 (Weighted) & {metrics['f1_weighted']:.2f} \\\\
Precision (Macro) & {metrics['precision_macro']:.2f} \\\\
Recall (Macro) & {metrics['recall_macro']:.2f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""

    # Table 2: Per-Class Results
    latex_table2 = """
\\begin{table}[h]
\\centering
\\caption{Per-Class Performance}
\\label{tab:baseline_per_class}
\\begin{tabular}{lcccc}
\\hline
Class & Accuracy ($\\%$) & Precision ($\\%$) & Recall ($\\%$) & F1 ($\\%$) \\\\
\\hline
"""

    for i, name in enumerate(metrics['label_names']):
        latex_table2 += (f"{name.capitalize():<15} & "
                        f"{metrics['accuracies_per_class'][i]:.2f} & "
                        f"{metrics['precision_per_class'][i]:.2f} & "
                        f"{metrics['recall_per_class'][i]:.2f} & "
                        f"{metrics['f1_per_class'][i]:.2f} \\\\\n")

    latex_table2 += """\\hline
\\end{tabular}
\\end{table}
"""

    # Table 3: Confusion Matrix
    cm = metrics['confusion_matrix']
    latex_table3 = """
\\begin{table}[h]
\\centering
\\caption{Confusion Matrix}
\\label{tab:baseline_confusion}
\\begin{tabular}{l|cccccc}
\\hline
 & \\multicolumn{6}{c}{Predicted} \\\\
\\cline{2-7}
Actual & T & MT & HT & BT & F & PF \\\\
\\hline
"""

    label_short = ['T', 'MT', 'HT', 'BT', 'F', 'PF']
    for i in range(6):
        row = " & ".join([str(x) for x in cm[i]])
        latex_table3 += f"{label_short[i]:<8} & {row} \\\\\n"

    latex_table3 += """\\hline
\\end{tabular}
\\end{table}
"""

    # 保存LaTeX表格
    tables_dir = os.path.join(output_dir, 'latex_tables')
    os.makedirs(tables_dir, exist_ok=True)

    with open(os.path.join(tables_dir, 'table_overall.tex'), 'w') as f:
        f.write(latex_table1)
    print(f"✓ Table 1 (Overall) saved to: {tables_dir}/table_overall.tex")

    with open(os.path.join(tables_dir, 'table_per_class.tex'), 'w') as f:
        f.write(latex_table2)
    print(f"✓ Table 2 (Per-Class) saved to: {tables_dir}/table_per_class.tex")

    with open(os.path.join(tables_dir, 'table_confusion.tex'), 'w') as f:
        f.write(latex_table3)
    print(f"✓ Table 3 (Confusion) saved to: {tables_dir}/table_confusion.tex")


def save_metrics_to_json(metrics, output_path):
    """保存指标到JSON文件"""
    # 转换numpy类型为python类型
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            metrics_json[key] = float(value)
        else:
            metrics_json[key] = value

    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)

    print(f"✓ Metrics saved to: {output_path}")


def analyze_checkpoint(checkpoint_dir, model_class=None, test_loader=None, device='cpu'):
    """
    分析训练好的checkpoint

    Args:
        checkpoint_dir: checkpoint目录
        model_class: 模型类（如果需要重新预测）
        test_loader: 测试数据加载器（如果需要重新预测）
        device: 设备
    """
    print("=" * 80)
    print("ANALYZING BASELINE MLP RESULTS")
    print("=" * 80)

    # 加载结果
    history, checkpoint = load_results(checkpoint_dir)

    print(f"\nLoaded checkpoint from: {checkpoint_dir}")
    print(f"Best epoch: {checkpoint['epoch'] + 1}")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")

    # 获取测试集指标
    test_metrics = history['test_metrics']

    # 计算详细指标
    metrics = compute_detailed_metrics(
        test_metrics.get('predictions', []),
        test_metrics.get('labels', [])
    )

    # 打印结果
    print_detailed_results(metrics)

    # 创建输出目录
    output_dir = os.path.join(checkpoint_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # 绘制混淆矩阵
    plot_confusion_matrix(metrics, os.path.join(output_dir, 'confusion_matrix.png'))

    # 生成LaTeX表格
    generate_latex_tables(metrics, checkpoint_dir)

    # 保存指标
    save_metrics_to_json(metrics, os.path.join(output_dir, 'metrics.json'))

    # 保存训练历史曲线
    plot_training_curves(history, output_dir)

    print("\n" + "=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80)

    return metrics


def plot_training_curves(history, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss曲线
    axes[0].plot(history['train_losses'], label='Train Loss', marker='o', markersize=3)
    axes[0].plot(history['val_losses'], label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[1].plot(history['train_accs'], label='Train Acc', marker='o', markersize=3)
    axes[1].plot(history['val_accs'], label='Val Acc', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {output_dir}/training_curves.png")
    plt.close()


if __name__ == "__main__":
    import sys

    checkpoint_dir = './checkpoints/baseline_mlp'

    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]

    # 分析结果
    metrics = analyze_checkpoint(checkpoint_dir)

    print("\n✅ Analysis complete!")
    print("\nGenerated files:")
    print(f"  - confusion_matrix.png")
    print(f"  - training_curves.png")
    print(f"  - metrics.json")
    print(f"  - latex_tables/")
