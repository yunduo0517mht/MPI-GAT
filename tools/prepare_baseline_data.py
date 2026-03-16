#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
准备纯净的baseline数据集
从 Liar_clean 提取 statement + label
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_clean_baseline(input_file, output_file):
    """
    从 Liar_clean 的 TSV 文件提取纯净的 baseline 数据

    Args:
        input_file: data/Liar_clean/train.tsv
        output_file: data/baseline/train_clean.csv
    """
    # 读取数据
    df = pd.read_csv(input_file, sep='\t')

    # 只保留 statement + label
    df_clean = df[['statement', 'label']].copy()

    # 将标签从字符串转为数字
    label_mapping = {
        'true': 0,
        'mostly-true': 1,
        'half-true': 2,
        'mostly-false': 3,
        'false': 4,
        'pants-fire': 5
    }

    df_clean['label_id'] = df_clean['label'].map(label_mapping)

    # 检查缺失值
    print(f"原始数据: {len(df)} 条")
    print(f"处理后: {len(df_clean)} 条")
    print(f"缺失值: {df_clean.isnull().sum().sum()}")
    print()

    # 显示类别分布
    print("标签分布:")
    print(df_clean['label'].value_counts().sort_index())
    print()

    # 显示示例
    print("数据示例:")
    print(df_clean.head(3))
    print()

    # 保存为CSV（方便后续使用）
    df_clean.to_csv(output_file, index=False)
    print(f"✓ 已保存到: {output_file}")

    return df_clean

if __name__ == '__main__':
    print("="*60)
    print("准备纯净 Baseline 数据集")
    print("="*60)
    print()

    # 处理 train/val/test 三个文件
    splits = [
        ('data/Liar_clean/train.tsv', 'data/baseline/train_clean.csv'),
        ('data/Liar_clean/valid.tsv', 'data/baseline/valid_clean.csv'),
        ('data/Liar_clean/test.tsv', 'data/baseline/test_clean.csv'),
    ]

    for input_file, output_file in splits:
        print(f"\n处理: {input_file}")
        prepare_clean_baseline(input_file, output_file)

    print("\n" + "="*60)
    print("完成！所有文件已保存到 data/baseline/")
    print("="*60)
