"""
分析训练过拟合情况
"""
import torch
import matplotlib.pyplot as plt
import json
import os

def analyze_training_history(checkpoint_dir):
    """分析训练历史"""

    history_file = os.path.join(checkpoint_dir, 'training_history.pth')
    results_file = os.path.join(checkpoint_dir, 'results_summary.json')

    if not os.path.exists(history_file):
        print(f"History file not found: {history_file}")
        return

    # 加载训练历史
    history = torch.load(history_file)

    train_losses = history['train_losses']
    train_accs = history['train_accs']
    val_losses = history['val_losses']
    val_accs = history['val_accs']

    # 加载测试结果
    with open(results_file, 'r') as f:
        results = json.load(f)

    print("=" * 80)
    print("Training History Analysis")
    print("=" * 80)

    print(f"\nCheckpoint: {checkpoint_dir}")
    print(f"Total epochs: {len(train_losses)}")
    print(f"Best epoch: {history['best_epoch']}")

    print("\n" + "-" * 80)
    print("Performance Summary")
    print("-" * 80)

    # 最终训练集性能
    final_train_loss = train_losses[-1]
    final_train_acc = train_accs[-1]

    # 最佳验证集性能
    best_val_loss = history['best_val_loss']
    best_val_acc = history['best_val_acc']

    # 测试集性能
    test_acc = results['test_accuracy']

    print(f"\nFinal Training:")
    print(f"  Loss: {final_train_loss:.4f}")
    print(f"  Acc:  {final_train_acc:.2f}%")

    print(f"\nBest Validation (epoch {history['best_epoch']}):")
    print(f"  Loss: {best_val_loss:.4f}")
    print(f"  Acc:  {best_val_acc:.2f}%")

    print(f"\nTest Set:")
    print(f"  Acc:  {test_acc:.2f}%")

    print("\n" + "-" * 80)
    print("Overfitting Analysis")
    print("-" * 80)

    # 计算过拟合指标
    train_val_gap = final_train_acc - best_val_acc
    val_test_gap = best_val_acc - test_acc
    train_test_gap = final_train_acc - test_acc

    print(f"\nAccuracy Gaps:")
    print(f"  Train - Val:  {train_val_gap:+.2f}%")
    print(f"  Val - Test:   {val_test_gap:+.2f}%")
    print(f"  Train - Test: {train_test_gap:+.2f}%")

    # 判断过拟合类型
    print("\nDiagnosis:")
    if train_val_gap > 20:
        print("  [SEVERE] Training set overfitting (Train >> Val)")
    elif train_val_gap > 10:
        print("  [MODERATE] Training set overfitting (Train > Val)")
    else:
        print("  [OK] Training-Validation gap is acceptable")

    if val_test_gap > 20:
        print("  [SEVERE] Validation-Test distribution mismatch (Val >> Test)")
        print("  -> This suggests the validation set is NOT representative of test set")
    elif val_test_gap > 10:
        print("  [MODERATE] Validation-Test gap")
    else:
        print("  [OK] Validation-Test gap is acceptable")

    # 检查训练曲线趋势
    print("\n" + "-" * 80)
    print("Training Curve Analysis")
    print("-" * 80)

    # 前5个epoch
    print(f"\nFirst 5 epochs:")
    for i in range(min(5, len(train_accs))):
        print(f"  Epoch {i+1}: Train={train_accs[i]:.2f}%, Val={val_accs[i]:.2f}%")

    # 最后5个epoch
    print(f"\nLast 5 epochs:")
    start_idx = max(0, len(train_accs) - 5)
    for i in range(start_idx, len(train_accs)):
        print(f"  Epoch {i+1}: Train={train_accs[i]:.2f}%, Val={val_accs[i]:.2f}%")

    # 检查验证集是否还在提升
    if len(val_accs) >= 10:
        early_val_acc = sum(val_accs[:5]) / 5
        late_val_acc = sum(val_accs[-5:]) / 5
        improvement = late_val_acc - early_val_acc

        print(f"\nValidation improvement (first 5 vs last 5 epochs): {improvement:+.2f}%")
        if improvement < 1:
            print("  -> Validation accuracy plateaued early")

    print("\n" + "=" * 80)


def main():
    # 分析所有运行
    base_dir = "checkpoints/gnn/gcn"

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # 找到所有 run_X 目录
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    run_dirs.sort()

    if not run_dirs:
        print(f"No run directories found in {base_dir}")
        return

    print(f"Found {len(run_dirs)} runs\n")

    for run_dir in run_dirs:
        checkpoint_dir = os.path.join(base_dir, run_dir)
        analyze_training_history(checkpoint_dir)
        print("\n")


if __name__ == "__main__":
    main()
