"""
按元路径拆分异构图为多个子图
基于 5 条元路径，将完整的异构图拆分为 5 个语义特定的子图
"""
import json
from pathlib import Path
from collections import defaultdict


class MetapathGraphSplitter:
    """基于元路径的图拆分器"""

    def __init__(self, input_file: str):
        """
        初始化拆分器

        Args:
            input_file: 向量化后的异构图 JSON 文件
        """
        self.input_file = input_file
        self.graph_data = None
        self.stats = {}

        # 定义 5 条元路径
        self.metapaths = [
            {
                "name": "news_author_news",
                "description": "News-Author-News: 捕获作者写作风格、立场",
                "node_types": ["news", "author"],
                "edge_types": ["publish"]
            },
            {
                "name": "news_comment_news",
                "description": "News-Comment-News: 捕获传播效果、话题延续",
                "node_types": ["news", "comment"],
                "edge_types": ["contain"]
            },
            {
                "name": "news_comment_author",
                "description": "News-Comment-Author: 读者对作者的反馈",
                "node_types": ["news", "comment", "author"],
                "edge_types": ["contain", "feedback"]
            },
            {
                "name": "comment_author_comment",
                "description": "Comment-Author-Comment: 作者-读者互动",
                "node_types": ["comment", "author"],
                "edge_types": ["author_reply"]
            },
            {
                "name": "comment_comment_comment",
                "description": "Comment-Comment-Comment: 读者间互动",
                "node_types": ["comment"],
                "edge_types": ["reply"]
            }
        ]

    def load_graph(self):
        """加载异构图数据"""
        print(f"Loading heterograph from {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.graph_data = json.load(f)

        print(f"  Author nodes: {len(self.graph_data.get('author_nodes', {}))}")
        print(f"  News nodes: {len(self.graph_data.get('news_nodes', {}))}")
        print(f"  Comment nodes: {len(self.graph_data.get('comment_nodes', {}))}")
        print(f"  Total edges: {len(self.graph_data.get('edges', []))}")

    def split_by_metapath(self, output_dir: str):
        """
        按元路径拆分图

        Args:
            output_dir: 输出目录
        """
        print("\n" + "=" * 60)
        print("Splitting Heterograph by Metapaths")
        print("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # 为每条元路径创建子图
        for metapath in self.metapaths:
            print(f"\nProcessing: {metapath['name']}")
            print(f"  Description: {metapath['description']}")

            # 提取子图
            subgraph = self.extract_subgraph(metapath)

            # 保存为 PyG 格式
            output_file = output_path / f"liar_test_{metapath['name']}.pt"
            self.save_pyg_format(subgraph, str(output_file), metapath)

            # 打印统计
            self.stats[metapath['name']] = {
                "num_nodes": sum(len(v) for v in subgraph["nodes"].values()),
                "num_edges": len(subgraph["edges"])
            }

            print(f"  [OK] Saved to {output_file}")
            print(f"  Nodes: {self.stats[metapath['name']]['num_nodes']}")
            print(f"  Edges: {self.stats[metapath['name']]['num_edges']}")

        print("\n" + "=" * 60)
        print("Splitting Completed!")
        print("=" * 60)

    def extract_subgraph(self, metapath: dict) -> dict:
        """
        提取元路径对应的子图

        Args:
            metapath: 元路径定义

        Returns:
            子图数据
        """
        # 1. 提取相关节点
        nodes = defaultdict(dict)
        node_ids = set()

        for node_type in metapath["node_types"]:
            type_key = f"{node_type}_nodes"
            if type_key in self.graph_data:
                for node_id, node_data in self.graph_data[type_key].items():
                    nodes[node_type][node_id] = node_data
                    node_ids.add(node_id)

        # 2. 提取相关边
        edges = []
        for edge in self.graph_data.get("edges", []):
            edge_type = edge.get("metadata", {}).get("edge_type", "")

            # 只保留元路径中定义的边类型
            if edge_type in metapath["edge_types"]:
                source_id = edge["source_id"]
                target_id = edge["target_id"]

                # 只保留两端节点都在子图中的边
                if source_id in node_ids and target_id in node_ids:
                    edges.append(edge)

        return {
            "nodes": nodes,
            "edges": edges,
            "metapath": metapath
        }

    def save_pyg_format(self, subgraph: dict, output_file: str, metapath: dict):
        """
        保存为 PyTorch Geometric 格式

        Args:
            subgraph: 子图数据
            output_file: 输出文件路径
            metapath: 元路径定义
        """
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            print("  [ERROR] torch_geometric not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'torch_geometric'])
            from torch_geometric.data import HeteroData

        import torch
        data = HeteroData()

        # 1. 构建节点特征矩阵
        node_id_to_idx = {}
        current_idx = 0

        # 为每种节点类型创建索引映射
        for node_type in metapath["node_types"]:
            type_nodes = subgraph["nodes"].get(node_type, {})
            for node_id in type_nodes:
                node_id_to_idx[node_id] = current_idx
                current_idx += 1

        # 为每种节点类型构建特征矩阵
        for node_type in metapath["node_types"]:
            type_nodes = subgraph["nodes"].get(node_type, {})

            if len(type_nodes) == 0:
                continue

            # 提取特征
            features_list = []
            for node_id, node_data in type_nodes.items():
                if "features" in node_data and node_data["features"]:
                    features_list.append(node_data["features"])
                else:
                    # 如果没有特征，用零向量
                    features_list.append([0.0] * 1024)

            # 转换为张量
            x = torch.tensor(features_list, dtype=torch.float32)
            data[node_type].x = x

            # 如果是新闻节点，添加标签
            if node_type == "news":
                labels = []
                for node_id, node_data in type_nodes.items():
                    label = node_data.get("label", 0)
                    labels.append(label)
                data[node_type].y = torch.tensor(labels, dtype=torch.long)

        # 2. 构建边索引
        edge_indices = defaultdict(list)

        for edge in subgraph["edges"]:
            source_id = edge["source_id"]
            target_id = edge["target_id"]
            edge_type = edge.get("metadata", {}).get("edge_type", "")

            # 确定源节点和目标节点的类型
            source_type = self.get_node_type(source_id)
            target_type = self.get_node_type(target_id)

            if source_type is None or target_type is None:
                continue

            # 获取索引
            source_idx = node_id_to_idx.get(source_id, -1)
            target_idx = node_id_to_idx.get(target_id, -1)

            if source_idx >= 0 and target_idx >= 0:
                edge_indices[(source_type, edge_type, target_type)].append((source_idx, target_idx))

        # 转换为张量
        for key, indices in edge_indices.items():
            if len(indices) > 0:
                edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
                data[key].edge_index = edge_index

        # 保存
        torch.save(data, output_file)

    def get_node_type(self, node_id: str):
        """
        获取节点类型

        Args:
            node_id: 节点ID

        Returns:
            节点类型
        """
        # 检查每种节点类型
        for node_type in ["author", "news", "comment"]:
            type_key = f"{node_type}_nodes"
            if type_key in self.graph_data and node_id in self.graph_data[type_key]:
                return node_type
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Split heterograph by metapaths')
    parser.add_argument(
        '--input',
        type=str,
        default='data/liar_test_heterograph_vectorized.json',
        help='Input vectorized heterograph JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/metapath_subgraphs',
        help='Output directory for subgraphs'
    )

    args = parser.parse_args()

    # 创建拆分器
    splitter = MetapathGraphSplitter(args.input)

    # 加载图
    splitter.load_graph()

    # 拆分
    splitter.split_by_metapath(args.output_dir)


if __name__ == '__main__':
    main()
