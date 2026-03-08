"""
从 LLM 生成的 JSON 文件构建异构图
将多个新闻-评论网络文件合并为一个大的异构图
"""
import json
import os
from pathlib import Path
from typing import Dict, List
from heterograph_structures import (
    HeteroGraph, AuthorNode, NewsNode, CommentNode, Edge, NodeType
)


class HeteroGraphBuilder:
    """异构图构建器"""

    def __init__(self):
        self.graph = HeteroGraph(graph_id="liar_complete_graph")
        self.global_authors = {}  # {speaker_id: AuthorNode}
        self.stats = {
            "total_files": 0,
            "total_news": 0,
            "total_comments": 0,
            "total_author_replies": 0,
            "total_edges": 0
        }

    def process_json_file(self, filepath: str):
        """
        处理单个 JSON 文件，将其内容添加到全局图中

        Args:
            filepath: JSON 文件路径
        """
        self.stats["total_files"] += 1

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 处理作者节点（全局唯一）
        speaker = data["news_speaker"]
        if speaker not in self.global_authors:
            author_node = AuthorNode(
                node_id=speaker,
                author_id=speaker,
                name=data.get("news_speaker_name", speaker),
                metadata={
                    "title": data.get("news_speaker_title", ""),
                    "party": data.get("news_party", "")
                }
            )
            self.global_authors[speaker] = author_node
            self.graph.add_node(author_node)

        # 2. 创建新闻节点
        news_id = data["news_id"]
        news_node = NewsNode(
            node_id=news_id,
            news_id=news_id,
            author_id=speaker,
            text=data["news_statement"],
            label=self._convert_label(data["news_label"]),
            metadata={
                "speaker": speaker,
                "title": data.get("news_speaker_title", ""),
                "party": data.get("news_party", ""),
                "context": data.get("news_context", "")
            }
        )
        self.graph.add_node(news_node)
        self.stats["total_news"] += 1

        # 3. 添加边: 作者 → 新闻
        self.graph.add_edge(Edge(
            source_id=speaker,
            target_id=news_id,
            metadata={"edge_type": "publish"}
        ))
        self.stats["total_edges"] += 1

        # 4. 处理评论节点
        comment_id_map = {}  # 用于记录评论ID映射
        for comment in data["comments"]:
            comment_id = comment["id"]
            comment_id_map[comment_id] = comment_id

            # 创建评论节点
            comment_node = CommentNode(
                node_id=comment_id,
                comment_id=comment_id,
                news_id=news_id,
                author_id=f"reader_{news_id}_{comment_id}",  # 虚拟读者ID
                text=comment["content"],
                metadata={
                    "likes": comment.get("likes", 0),
                    "is_author_reply": False
                }
            )
            self.graph.add_node(comment_node)
            self.stats["total_comments"] += 1

            # 添加边
            parent_id = comment.get("parent_id")
            if parent_id is None or parent_id == "null":
                # 顶级评论: 新闻 → 评论
                self.graph.add_edge(Edge(
                    source_id=news_id,
                    target_id=comment_id,
                    metadata={"edge_type": "contain"}
                ))
            else:
                # 回复评论: 父评论 → 子评论（单向）
                self.graph.add_edge(Edge(
                    source_id=parent_id,
                    target_id=comment_id,
                    metadata={"edge_type": "reply"}
                ))
            self.stats["total_edges"] += 1

        # 5. 处理作者回复
        for idx, reply in enumerate(data["author_replies"]):
            reply_id = f"author_reply_{news_id}_{idx}"

            reply_node = CommentNode(
                node_id=reply_id,
                comment_id=reply_id,
                news_id=news_id,
                author_id=speaker,  # 作者的回复，author_id 指向发言人
                text=reply["content"],
                metadata={
                    "likes": reply.get("likes", 0),
                    "is_author_reply": True
                }
            )
            self.graph.add_node(reply_node)
            self.stats["total_author_replies"] += 1

            # 添加边: 父评论 → 作者回复
            parent_id = reply["parent_id"]
            self.graph.add_edge(Edge(
                source_id=parent_id,
                target_id=reply_id,
                metadata={"edge_type": "author_reply"}
            ))
            self.stats["total_edges"] += 1

            # 添加边: 作者回复 → 作者（反馈给作者）
            self.graph.add_edge(Edge(
                source_id=reply_id,
                target_id=speaker,
                metadata={"edge_type": "feedback"}
            ))
            self.stats["total_edges"] += 1

        print(f"[OK] Processed: {filepath}")
        print(f"  - News: {news_id}")
        print(f"  - Speaker: {speaker}")
        print(f"  - Comments: {len(data['comments'])}")
        print(f"  - Author replies: {len(data['author_replies'])}")

    def process_directory(self, dirpath: str, pattern: str = "*.json"):
        """
        处理目录下的所有 JSON 文件

        Args:
            dirpath: 目录路径
            pattern: 文件匹配模式（默认 *.json）
        """
        dir_path = Path(dirpath)
        json_files = list(dir_path.glob(pattern))

        print(f"=" * 60)
        print(f"开始处理目录: {dirpath}")
        print(f"找到 {len(json_files)} 个 JSON 文件")
        print(f"=" * 60)

        for json_file in sorted(json_files):
            try:
                self.process_json_file(str(json_file))
            except Exception as e:
                print(f"[FAIL] Processing failed: {json_file}")
                print(f"  Error: {e}")

        # 定义元路径
        self.graph.define_metapaths()

        # 打印统计信息
        self.print_statistics()

    def print_statistics(self):
        """打印图的统计信息"""
        graph_stats = self.graph.get_statistics()

        print(f"\n{'='*60}")
        print("异构图构建完成！")
        print(f"{'='*60}")
        print(f"处理的文件数: {self.stats['total_files']}")
        print(f"\n节点统计:")
        print(f"  - 作者节点: {graph_stats['num_author_nodes']}")
        print(f"  - 新闻节点: {graph_stats['num_news_nodes']}")
        print(f"  - 评论节点: {graph_stats['num_comment_nodes']}")
        print(f"  - 总节点数: {graph_stats['total_nodes']}")
        print(f"\n详细统计:")
        print(f"  - 普通评论: {self.stats['total_comments']}")
        print(f"  - 作者回复: {self.stats['total_author_replies']}")
        print(f"  - 总评论数: {self.stats['total_comments'] + self.stats['total_author_replies']}")
        print(f"\n边统计:")
        print(f"  - 总边数: {self.stats['total_edges']}")
        print(f"\n元路径:")
        print(f"  - 定义了 {len(self.graph.metapaths)} 条元路径")

    def save_graph(self, output_file: str):
        """
        保存异构图到文件

        Args:
            output_file: 输出文件路径（.json 格式）
        """
        self.graph.save_to_json(output_file)
        print(f"\n[OK] Heterograph saved to: {output_file}")

    def _convert_label(self, label: str) -> int:
        """
        将标签转换为数值（保留原始6类）

        Args:
            label: 原始标签字符串

        Returns:
            数值标签 (0-5)
        """
        label_map = {
            'true': 0,
            'mostly-true': 1,
            'half-true': 2,
            'barely-true': 3,
            'false': 4,
            'pants-fire': 5
        }
        return label_map.get(label.lower(), 2)  # 默认 half-true


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='从 JSON 文件构建异构图')
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='包含 JSON 文件的输入目录'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/liar_heterograph.json',
        help='输出文件路径（默认: data/liar_heterograph.json）'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.json',
        help='JSON 文件匹配模式（默认: *.json）'
    )

    args = parser.parse_args()

    # 构建异构图
    builder = HeteroGraphBuilder()
    builder.process_directory(args.input_dir, args.pattern)

    # 保存图
    builder.save_graph(args.output_file)


if __name__ == '__main__':
    main()
