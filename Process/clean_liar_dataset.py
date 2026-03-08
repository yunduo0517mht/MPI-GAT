"""
LIAR 数据集清洗脚本
按照方案B（平衡方案）清洗数据，保留核心字段用于LLM生成评论
"""
import pandas as pd
import os
from pathlib import Path


def load_liar_data(filepath):
    """
    加载LIAR数据集的TSV文件
    """
    columns = [
        'id', 'label', 'statement', 'subjects', 'speaker',
        'job_title', 'state', 'party',
        'barely_true', 'false', 'half_true', 'mostly_true', 'pants_fire', 'context'
    ]
    df = pd.read_csv(filepath, sep='\t', header=None, names=columns)
    return df


def clean_single_dataset(input_file, output_file):
    """
    清洗单个数据集文件

    保留字段（方案B - 平衡方案）:
    - statement: 新闻内容
    - speaker: 发言人
    - label: 标签
    - subjects: 主题
    - job_title: 职位
    - party: 党派
    - context: 场合

    去掉字段:
    - id: 原始JSON文件名（无用）
    - state: 州信息（对评论生成影响较小）
    - history counts: 历史统计（5个字段，LLM难以直接利用）
    """
    print(f"正在处理: {input_file}")

    # 加载数据
    df = load_liar_data(input_file)
    print(f"  原始数据量: {len(df)} 条")

    # 选择保留的字段
    keep_columns = [
        'statement', 'speaker', 'label', 'subjects',
        'job_title', 'party', 'context'
    ]

    df_clean = df[keep_columns].copy()

    # 处理缺失值
    print(f"  处理缺失值...")

    # 检查各字段的缺失情况
    missing_stats = {}
    for col in df_clean.columns:
        missing_count = df_clean[col].isna().sum() + (df_clean[col] == '').sum()
        if missing_count > 0:
            missing_stats[col] = missing_count
            print(f"    {col}: {missing_count} 条缺失")

    # 填充缺失值
    df_clean['speaker'] = df_clean['speaker'].fillna('Unknown')
    df_clean['job_title'] = df_clean['job_title'].fillna('Unknown')
    df_clean['party'] = df_clean['party'].fillna('unknown')
    df_clean['subjects'] = df_clean['subjects'].fillna('general')
    df_clean['context'] = df_clean['context'].fillna('unknown')

    # 替换空字符串
    df_clean['speaker'] = df_clean['speaker'].replace('', 'Unknown')
    df_clean['job_title'] = df_clean['job_title'].replace('', 'Unknown')
    df_clean['party'] = df_clean['party'].replace('', 'unknown')
    df_clean['subjects'] = df_clean['subjects'].replace('', 'general')
    df_clean['context'] = df_clean['context'].replace('', 'unknown')

    # 去除 statement 中的前后空格
    df_clean['statement'] = df_clean['statement'].str.strip()

    # 检查是否有空的 statement（这些是无效样本，应该删除）
    empty_statement = df_clean['statement'].isna() | (df_clean['statement'] == '')
    if empty_statement.sum() > 0:
        print(f"  警告: 发现 {empty_statement.sum()} 条空的 statement，将被删除")
        df_clean = df_clean[~empty_statement]

    print(f"  清洗后数据量: {len(df_clean)} 条")

    # 保存清洗后的数据
    df_clean.to_csv(output_file, index=False, sep='\t')
    print(f"  已保存到: {output_file}")

    # 显示统计信息
    print(f"\n  标签分布:")
    label_counts = df_clean['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"    {label:15s}: {count:5d} ({count/len(df_clean)*100:.1f}%)")

    return df_clean


def main():
    """
    主函数：清洗所有LIAR数据集文件
    """
    # 路径配置
    base_dir = Path('data')
    input_dir = base_dir / 'Liar'
    output_dir = base_dir / 'Liar_clean'

    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 要处理的文件
    files_to_process = [
        ('train.tsv', 'train.tsv'),
        ('valid.tsv', 'valid.tsv'),
        ('test.tsv', 'test.tsv')
    ]

    # 处理每个文件
    summaries = {}
    for input_file, output_file in files_to_process:
        input_path = input_dir / input_file
        output_path = output_dir / output_file

        if input_path.exists():
            print(f"\n{'='*60}")
            df_clean = clean_single_dataset(input_path, output_path)
            summaries[input_file] = df_clean
        else:
            print(f"\n警告: 文件不存在 {input_path}")

    # 总体统计
    print(f"\n{'='*60}")
    print("总体统计:")
    print(f"{'='*60}")

    total_train = len(summaries.get('train.tsv', []))
    total_valid = len(summaries.get('valid.tsv', []))
    total_test = len(summaries.get('test.tsv', []))
    total = total_train + total_valid + total_test

    print(f"训练集: {total_train:,} 条")
    print(f"验证集: {total_valid:,} 条")
    print(f"测试集: {total_test:,} 条")
    print(f"总计:   {total:,} 条")

    print(f"\n清洗完成！")
    print(f"清洗后的数据保存在: {output_dir}")


if __name__ == '__main__':
    main()
