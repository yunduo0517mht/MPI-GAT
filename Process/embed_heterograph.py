"""
异构图节点文本向量化脚本
使用公司自部署的 embedding 模型将节点文本转换为向量
"""
import json
import requests
import time
from pathlib import Path
from typing import List, Dict
import numpy as np


class HeteroGraphEmbedder:
    """异构图向量化器"""

    def __init__(self, api_url: str = "http://pypi.gamework.ai:28001", model_name: str = "qwen3-embedding-0.6b"):
        """
        初始化向量化器

        Args:
            api_url: Embedding API 地址
            model_name: 模型名称
        """
        self.api_url = api_url
        self.model_name = model_name
        self.stats = {
            "total_nodes": 0,
            "success": 0,
            "failed": 0
        }

    def call_embedding_api(self, text: str) -> List[float]:
        """
        调用 embedding API

        Args:
            text: 输入文本

        Returns:
            1024 维向量
        """
        # TODO: 根据实际 API 格式调整
        # 这里提供两种常见格式，请根据实际情况选择或修改

        try:
            # 方式 A: OpenAI 兼容格式
            response = requests.post(
                f"{self.api_url}/v1/embeddings",
                json={
                    "model": self.model_name,
                    "input": text
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            # OpenAI 格式返回
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]

            # 方式 B: 自定义格式（如果上面不行，试试这个）
            # response = requests.post(
            #     self.api_url,
            #     json={"text": text, "model": self.model_name},
            #     timeout=30
            # )
            # result = response.json()
            # return result.get("embedding", [])

            # 如果都不行，请查看您的 API 文档，修改这里
            raise ValueError("Unknown API response format")

        except Exception as e:
            print(f"  [ERROR] API call failed: {e}")
            return None

    def prepare_node_texts(self, graph_data: Dict) -> Dict[str, List[str]]:
        """
        准备所有节点的文本

        Args:
            graph_data: 异构图数据

        Returns:
            {node_type: [(node_id, text), ...]}
        """
        texts_by_type = {
            "news": [],
            "author": [],
            "comment": []
        }

        # 新闻节点：直接使用 text 字段
        for node_id, node_data in graph_data.get("news_nodes", {}).items():
            text = node_data.get("text", "")
            texts_by_type["news"].append((node_id, text))

        # 评论节点：直接使用 text 字段
        for node_id, node_data in graph_data.get("comment_nodes", {}).items():
            text = node_data.get("text", "")
            texts_by_type["comment"].append((node_id, text))

        # 作者节点：拼接多个字段
        for node_id, node_data in graph_data.get("author_nodes", {}).items():
            parts = [
                node_data.get("name", ""),
                node_data.get("metadata", {}).get("party", ""),
                node_data.get("metadata", {}).get("title", "")
            ]
            text = " ".join([p for p in parts if p]).strip()
            texts_by_type["author"].append((node_id, text))

        return texts_by_type

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量向量化

        Args:
            texts: 文本列表
            batch_size: 批大小

        Returns:
            向量列表
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch_texts)} texts)")

            for text in batch_texts:
                # 调用 API
                embedding = self.call_embedding_api(text)

                if embedding is not None:
                    embeddings.append(embedding)
                    self.stats["success"] += 1
                else:
                    # 失败时用零向量填充
                    embeddings.append([0.0] * 1024)
                    self.stats["failed"] += 1

                # 避免请求过快
                time.sleep(0.1)

        return embeddings

    def embed_graph(self, input_file: str, output_file: str, batch_size: int = 32):
        """
        对整个异构图进行向量化

        Args:
            input_file: 输入的异构图 JSON 文件
            output_file: 输出的带向量的异构图 JSON 文件
            batch_size: 批大小
        """
        print("=" * 60)
        print("Heterograph Embedding")
        print("=" * 60)
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"API: {self.api_url}")
        print(f"Model: {self.model_name}")
        print("=" * 60)

        # 1. 加载异构图
        print("\n[1/4] Loading heterograph...")
        with open(input_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        total_nodes = (
            len(graph_data.get("author_nodes", {})) +
            len(graph_data.get("news_nodes", {})) +
            len(graph_data.get("comment_nodes", {}))
        )
        print(f"  Total nodes: {total_nodes}")
        self.stats["total_nodes"] = total_nodes

        # 2. 准备文本
        print("\n[2/4] Preparing node texts...")
        texts_by_type = self.prepare_node_texts(graph_data)

        for node_type, items in texts_by_type.items():
            print(f"  {node_type}: {len(items)} nodes")

        # 3. 批量向量化
        print("\n[3/4] Embedding nodes...")
        for node_type, items in texts_by_type.items():
            print(f"\n  Processing {node_type} nodes ({len(items)} nodes)...")

            # 提取文本
            node_ids = [item[0] for item in items]
            texts = [item[1] for item in items]

            # 批量向量化
            embeddings = self.batch_embed(texts, batch_size)

            # 更新图数据
            nodes_dict = graph_data.get(f"{node_type}_nodes", {})
            for node_id, embedding in zip(node_ids, embeddings):
                if node_id in nodes_dict:
                    nodes_dict[node_id]["features"] = embedding

        # 4. 保存结果
        print("\n[4/4] Saving embedded graph...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        print(f"  Saved to: {output_file}")

        # 打印统计
        print("\n" + "=" * 60)
        print("Embedding Completed!")
        print("=" * 60)
        print(f"Total nodes: {self.stats['total_nodes']}")
        print(f"Success: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {self.stats['success']/self.stats['total_nodes']*100:.1f}%")
        print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Embed heterograph nodes')
    parser.add_argument(
        '--input',
        type=str,
        default='data/liar_test_heterograph.json',
        help='Input heterograph JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/liar_test_heterograph_embedded.json',
        help='Output embedded heterograph JSON file'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://pypi.gamework.ai:28001',
        help='Embedding API URL'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='qwen3-embedding-0.6b',
        help='Model name'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding'
    )

    args = parser.parse_args()

    # 创建向量化器
    embedder = HeteroGraphEmbedder(
        api_url=args.api_url,
        model_name=args.model
    )

    # 执行向量化
    embedder.embed_graph(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
