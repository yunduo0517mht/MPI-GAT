"""
直接修改异构图中的新闻标签（从3类改回6类）
从原始JSON文件中读取标签信息
"""
import json
from pathlib import Path
from glob import glob


def fix_labels_from_original_json(
    heterograph_file: str,
    original_json_dir: str,
    output_file: str
):
    """
    从原始JSON文件中读取标签，修正异构图中的新闻标签

    Args:
        heterograph_file: 输入的异构图JSON文件
        original_json_dir: 原始JSON文件所在目录
        output_file: 输出的修正后的JSON文件
    """
    print("=" * 60)
    print("Fixing News Labels in Heterograph")
    print("=" * 60)

    # 1. 从原始JSON文件中读取所有标签
    print(f"\n[1/3] Reading labels from original JSON files...")
    print(f"  Directory: {original_json_dir}")

    label_mapping = {}  # {news_id: label_string}
    json_files = glob(str(Path(original_json_dir) / "*.json"))

    print(f"  Found {len(json_files)} JSON files")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            news_id = data.get("news_id")
            news_label = data.get("news_label")

            if news_id and news_label:
                label_mapping[news_id] = news_label
        except Exception as e:
            print(f"  [WARN] Failed to read {json_file}: {e}")

    print(f"  Loaded {len(label_mapping)} labels")

    # 2. 定义6类标签映射
    label_map_6class = {
        'true': 0,
        'mostly-true': 1,
        'half-true': 2,
        'barely-true': 3,
        'false': 4,
        'pants-fire': 5
    }

    # 3. 读取异构图
    print(f"\n[2/3] Loading heterograph...")
    print(f"  File: {heterograph_file}")

    with open(heterograph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    print(f"  Author nodes: {len(graph_data.get('author_nodes', {}))}")
    print(f"  News nodes: {len(graph_data.get('news_nodes', {}))}")
    print(f"  Comment nodes: {len(graph_data.get('comment_nodes', {}))}")

    # 4. 修正标签
    print(f"\n[3/3] Fixing news labels...")

    fixed_count = 0
    not_found_count = 0
    total_news = len(graph_data.get('news_nodes', {}))

    # 统计修正前后的标签分布
    old_label_dist = {}
    new_label_dist = {}

    for news_id, news_data in graph_data.get('news_nodes', {}).items():
        # 统计旧标签
        old_label = news_data.get('label', -1)
        old_label_dist[old_label] = old_label_dist.get(old_label, 0) + 1

        # 从映射表中获取原始标签字符串
        if news_id in label_mapping:
            original_label_str = label_mapping[news_id]
            new_label = label_map_6class.get(original_label_str, 2)

            # 更新标签
            if old_label != new_label:
                news_data['label'] = new_label
                fixed_count += 1

            # 统计新标签
            new_label_dist[new_label] = new_label_dist.get(new_label, 0) + 1
        else:
            not_found_count += 1
            print(f"  [WARN] Label not found for {news_id}")

    print(f"\n  Fixed: {fixed_count}/{total_news} news labels")
    print(f"  Not found: {not_found_count} news")

    # 打印标签分布
    print(f"\n  Old label distribution (3-class):")
    for label in sorted(old_label_dist.keys()):
        print(f"    {label}: {old_label_dist[label]}")

    print(f"\n  New label distribution (6-class):")
    label_names = {
        0: 'true',
        1: 'mostly-true',
        2: 'half-true',
        3: 'barely-true',
        4: 'false',
        5: 'pants-fire'
    }
    for label in sorted(new_label_dist.keys()):
        print(f"    {label} ({label_names.get(label, 'unknown')}): {new_label_dist[label]}")

    # 5. 保存
    print(f"\n[4/4] Saving fixed heterograph...")
    print(f"  Output: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("[OK] Label fixing completed!")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fix labels in heterograph from original JSON files')
    parser.add_argument(
        '--heterograph',
        type=str,
        default='data/liar_test_heterograph.json',
        help='Input heterograph JSON file'
    )
    parser.add_argument(
        '--original-dir',
        type=str,
        default='data/greated_Liar/test',
        help='Directory containing original JSON files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/liar_test_heterograph.json',
        help='Output heterograph JSON file (default: overwrite input)'
    )

    args = parser.parse_args()

    fix_labels_from_original_json(
        heterograph_file=args.heterograph,
        original_json_dir=args.original_dir,
        output_file=args.output
    )
