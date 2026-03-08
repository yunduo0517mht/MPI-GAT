"""
简单展示Baseline MLP结果（基于已有的训练指标）
"""

import torch
import json
import os


def show_results(checkpoint_dir='./checkpoints/baseline_mlp'):
    """展示结果"""

    print("=" * 80)
    print("BASELINE MLP - EXPERIMENTAL RESULTS")
    print("=" * 80)

    # 加载历史
    history = torch.load(os.path.join(checkpoint_dir, 'training_history.pth'))

    print("\n1. TRAINING SUMMARY")
    print("-" * 80)
    print(f"Best Epoch:          {history.get('best_epoch', 'N/A')}")
    print(f"Best Val Loss:       {history['best_val_loss']:.4f}")
    print(f"Best Val Accuracy:   {history['best_val_acc']:.2f}%")

    # 测试集结果
    test_metrics = history['test_metrics']

    print("\n2. TEST SET RESULTS")
    print("-" * 80)
    print(f"Accuracy:           {test_metrics['accuracy']:.2f}%")
    print(f"F1 (Macro):         {test_metrics['f1_macro']*100:.2f}%")
    print(f"F1 (Micro):         {test_metrics['f1_micro']*100:.2f}%")
    print(f"F1 (Weighted):      {test_metrics['f1_weighted']*100:.2f}%")
    print(f"Precision (Macro):  {test_metrics['precision_macro']*100:.2f}%")
    print(f"Recall (Macro):     {test_metrics['recall_macro']*100:.2f}%")

    print("\n3. PER-CLASS F1 SCORES")
    print("-" * 80)
    label_names = ['true', 'mostly-true', 'half-true',
                   'barely-true', 'false', 'pants-fire']

    print(f"{'Class':<15} {'F1 Score':>10}")
    print("-" * 80)
    for i, name in enumerate(label_names):
        print(f"{name:<15} {test_metrics['f1_per_class'][i]*100:>10.2f}%")

    print("\n4. FOR PAPER (LaTeX Format)")
    print("-" * 80)

    # Table 1: Overall Results
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

    # Table 2: Per-Class Results
    print("\n% Table: Per-Class F1 Scores")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Per-Class Performance}")
    print("\\begin{tabular}{lc}")
    print("\\hline")
    print("Class & F1 ($\\%$) \\\\")
    print("\\hline")
    for i, name in enumerate(label_names):
        print(f"{name.capitalize()} & {test_metrics['f1_per_class'][i]*100:.2f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # 保存到JSON
    output = {
        'model': 'Baseline_MLP',
        'stage': 1,
        'description': 'Text-only baseline',
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1_macro': float(test_metrics['f1_macro']*100),
        'test_f1_weighted': float(test_metrics['f1_weighted']*100),
        'test_precision_macro': float(test_metrics['precision_macro']*100),
        'test_recall_macro': float(test_metrics['recall_macro']*100),
        'per_class_f1': (test_metrics['f1_per_class'] * 100).tolist()
    }

    output_path = os.path.join(checkpoint_dir, 'results_summary.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("SUMMARY FOR ABLATION STUDY")
    print("=" * 80)
    print(f"Stage 1 (Baseline_MLP): {test_metrics['accuracy']:.2f}% accuracy")
    print("Input: News text only")
    print("=" * 80)

    return output


if __name__ == "__main__":
    show_results()
