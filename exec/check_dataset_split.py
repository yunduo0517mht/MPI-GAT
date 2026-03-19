"""
检查数据集划分是否合理

检查项：
1. 训练/验证/测试集的标签分布
2. 数据泄露（重复样本）
3. 说话人分布
4. 数据集统计信息
"""

import pandas as pd
import numpy as np
from collections import Counter
import os

# 标签映射
LABEL_MAP = {
    'pants-fire': 0,
    'false': 1,
    'barely-true': 2,
    'half-true': 3,
    'mostly-true': 4,
    'true': 5
}

LABEL_NAMES = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']


def load_dataset(file_path):
    """加载数据集"""
    df = pd.read_csv(file_path, sep='\t')
    return df


def check_label_distribution(train_df, val_df, test_df):
    """检查标签分布"""
    print("\n" + "=" * 80)
    print("1. 标签分布检查")
    print("=" * 80)

    for name, df in [('Train', train_df), ('Valid', val_df), ('Test', test_df)]:
        label_counts = df['label'].value_counts()
        total = len(df)

        print(f"\n{name} Set (Total: {total})")
        print("-" * 60)

        for label in LABEL_NAMES:
            count = label_counts.get(label, 0)
            pct = count / total * 100
            print(f"  {label:<15} {count:>6} ({pct:>5.2f}%)")

    # 计算分布差异
    print("\n" + "-" * 80)
    print("标签分布差异（与训练集对比）")
    print("-" * 80)

    train_dist = train_df['label'].value_counts(normalize=True).sort_index()
    val_dist = val_df['label'].value_counts(normalize=True).sort_index()
    test_dist = test_df['label'].value_counts(normalize=True).sort_index()

    print(f"\n{'Label':<15} {'Train %':>10} {'Valid %':>10} {'Test %':>10} {'Val Diff':>10} {'Test Diff':>10}")
    print("-" * 80)

    for label in LABEL_NAMES:
        train_pct = train_dist.get(label, 0) * 100
        val_pct = val_dist.get(label, 0) * 100
        test_pct = test_dist.get(label, 0) * 100
        val_diff = val_pct - train_pct
        test_diff = test_pct - train_pct

        print(f"{label:<15} {train_pct:>9.2f}% {val_pct:>9.2f}% {test_pct:>9.2f}% {val_diff:>+9.2f}% {test_diff:>+9.2f}%")


def check_data_leakage(train_df, val_df, test_df):
    """检查数据泄露（重复样本）"""
    print("\n" + "=" * 80)
    print("2. 数据泄露检查")
    print("=" * 80)

    # 检查 statement 重复
    train_statements = set(train_df['statement'].values)
    val_statements = set(val_df['statement'].values)
    test_statements = set(test_df['statement'].values)

    train_val_overlap = train_statements & val_statements
    train_test_overlap = train_statements & test_statements
    val_test_overlap = val_statements & test_statements

    print(f"\nStatement 重复检查:")
    print(f"  Train-Valid 重复: {len(train_val_overlap)} 条")
    print(f"  Train-Test 重复:  {len(train_test_overlap)} 条")
    print(f"  Valid-Test 重复:  {len(val_test_overlap)} 条")

    if len(train_test_overlap) > 0:
        print("\n  [WARNING] Train-Test overlap detected!")
        print(f"  Sample overlaps (first 3):")
        for i, stmt in enumerate(list(train_test_overlap)[:3]):
            print(f"    {i+1}. {stmt[:80]}...")

    if len(train_val_overlap) > 0:
        print("\n  [WARNING] Train-Valid overlap detected!")


def check_speaker_distribution(train_df, val_df, test_df):
    """检查说话人分布"""
    print("\n" + "=" * 80)
    print("3. 说话人分布检查")
    print("=" * 80)

    train_speakers = set(train_df['speaker'].values)
    val_speakers = set(val_df['speaker'].values)
    test_speakers = set(test_df['speaker'].values)

    print(f"\n说话人数量:")
    print(f"  Train: {len(train_speakers)} 人")
    print(f"  Valid: {len(val_speakers)} 人")
    print(f"  Test:  {len(test_speakers)} 人")

    # 检查说话人重叠
    train_val_speaker_overlap = train_speakers & val_speakers
    train_test_speaker_overlap = train_speakers & test_speakers
    val_test_speaker_overlap = val_speakers & test_speakers

    print(f"\n说话人重叠:")
    print(f"  Train-Valid: {len(train_val_speaker_overlap)} 人 ({len(train_val_speaker_overlap)/len(val_speakers)*100:.1f}% of valid)")
    print(f"  Train-Test:  {len(train_test_speaker_overlap)} 人 ({len(train_test_speaker_overlap)/len(test_speakers)*100:.1f}% of test)")
    print(f"  Valid-Test:  {len(val_test_speaker_overlap)} 人")

    # 检查是否有说话人只在测试集出现
    test_only_speakers = test_speakers - train_speakers
    val_only_speakers = val_speakers - train_speakers

    print(f"\n仅在测试/验证集出现的说话人:")
    print(f"  仅在 Valid: {len(val_only_speakers)} 人 ({len(val_only_speakers)/len(val_speakers)*100:.1f}%)")
    print(f"  仅在 Test:  {len(test_only_speakers)} 人 ({len(test_only_speakers)/len(test_speakers)*100:.1f}%)")

    if len(test_only_speakers) > 50:
        print(f"\n  [WARNING] {len(test_only_speakers)} speakers only appear in test set - may cause generalization issues!")


def check_party_distribution(train_df, val_df, test_df):
    """检查党派分布"""
    print("\n" + "=" * 80)
    print("4. 党派分布检查")
    print("=" * 80)

    print(f"\n{'Party':<15} {'Train %':>10} {'Valid %':>10} {'Test %':>10}")
    print("-" * 60)

    for name, df in [('Train', train_df), ('Valid', val_df), ('Test', test_df)]:
        party_counts = df['party'].value_counts()

    train_party = train_df['party'].value_counts(normalize=True)
    val_party = val_df['party'].value_counts(normalize=True)
    test_party = test_df['party'].value_counts(normalize=True)

    all_parties = set(train_party.index) | set(val_party.index) | set(test_party.index)

    for party in sorted(all_parties):
        train_pct = train_party.get(party, 0) * 100
        val_pct = val_party.get(party, 0) * 100
        test_pct = test_party.get(party, 0) * 100
        print(f"{party:<15} {train_pct:>9.2f}% {val_pct:>9.2f}% {test_pct:>9.2f}%")


def check_subject_distribution(train_df, val_df, test_df):
    """检查主题分布"""
    print("\n" + "=" * 80)
    print("5. 主题分布检查（Top 10）")
    print("=" * 80)

    # 统计主题
    def get_subjects(df):
        subjects = []
        for subj_str in df['subjects'].values:
            if pd.notna(subj_str):
                subjects.extend(subj_str.split(','))
        return Counter(subjects)

    train_subjects = get_subjects(train_df)
    val_subjects = get_subjects(val_df)
    test_subjects = get_subjects(test_df)

    print(f"\n{'Subject':<30} {'Train':>8} {'Valid':>8} {'Test':>8}")
    print("-" * 60)

    top_subjects = [s for s, _ in train_subjects.most_common(10)]

    for subj in top_subjects:
        train_cnt = train_subjects.get(subj, 0)
        val_cnt = val_subjects.get(subj, 0)
        test_cnt = test_subjects.get(subj, 0)
        print(f"{subj:<30} {train_cnt:>8} {val_cnt:>8} {test_cnt:>8}")


def main():
    data_dir = "data/Liar_clean"

    print("=" * 80)
    print("数据集划分检查")
    print("=" * 80)

    # 加载数据
    train_df = load_dataset(os.path.join(data_dir, "train.tsv"))
    val_df = load_dataset(os.path.join(data_dir, "valid.tsv"))
    test_df = load_dataset(os.path.join(data_dir, "test.tsv"))

    print(f"\n数据集大小:")
    print(f"  Train: {len(train_df):>6} 条")
    print(f"  Valid: {len(val_df):>6} 条")
    print(f"  Test:  {len(test_df):>6} 条")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df):>6} 条")

    # 执行检查
    check_label_distribution(train_df, val_df, test_df)
    check_data_leakage(train_df, val_df, test_df)
    check_speaker_distribution(train_df, val_df, test_df)
    check_party_distribution(train_df, val_df, test_df)
    check_subject_distribution(train_df, val_df, test_df)

    print("\n" + "=" * 80)
    print("检查完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
