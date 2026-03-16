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
import os
from tqdm import tqdm


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

    def call_embedding_api(self, text: str, max_retries: int = 3) -> List[float]:
        """
        调用 embedding API (带重试机制)

        Args:
            text: 输入文本
            max_retries: 最大重试次数

        Returns:
            向量
        """
        for attempt in range(max_retries):
            try:
                # OpenAI 兼容格式
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
                    embedding = result["data"][0]["embedding"]
                    self.stats["success"] += 1
                    return embedding

                raise ValueError("Unknown API response format")

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                    time.sleep(wait_time)
                    continue
                else:
                    self.stats["failed"] += 1
                    return None

    def process_single_file(self, input_file: str, output_file: str):
        """
        处理单个 comment_network 文件
        三节点设计:
        - 新闻节点: embedding 新闻原文
        - 作者节点: embedding speaker + party + context
        - 评论节点: embedding 评论内容 + likes 信息

        Args:
            input_file: 输入的 JSON 文件
            output_file: 输出的带向量的 JSON 文件
        """
        # 加载数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 新闻节点 - 只 embedding 新闻原文
        news_text = data.get("news_statement", "").strip()
        
        news_embedding = self.call_embedding_api(news_text)
        if news_embedding is None:
            news_embedding = [0.0] * 1024  # 失败时用零向量
        
        data["news_embedding"] = news_embedding
        time.sleep(0.05)

        # 2. 作者节点 - embedding speaker + speaker_title + party + context
        author_text_parts = [
            data.get("news_speaker", ""),
            data.get("news_speaker_title", ""),
            data.get("news_party", ""),
            data.get("news_context", "")
        ]
        author_text = " ".join([p for p in author_text_parts if p]).strip()
        
        author_embedding = self.call_embedding_api(author_text)
        if author_embedding is None:
            author_embedding = [0.0] * 1024
        
        data["author_embedding"] = author_embedding
        time.sleep(0.05)

        # 3. 评论节点 - embedding 内容 + likes
        for comment in data.get("comments", []):
            comment_content = comment.get("content", "")
            comment_likes = comment.get("likes", 0)
            # 拼接: 评论内容 + likes信息
            comment_text = f"{comment_content} [Likes: {comment_likes}]"
            
            embedding = self.call_embedding_api(comment_text)
            if embedding is None:
                embedding = [0.0] * 1024
            comment["embedding"] = embedding
            time.sleep(0.05)  # 避免请求过快

        # 4. 作者回复也作为评论节点处理
        for reply in data.get("author_replies", []):
            reply_content = reply.get("content", "")
            reply_likes = reply.get("likes", 0)
            reply_text = f"{reply_content} [Likes: {reply_likes}]"
            
            embedding = self.call_embedding_api(reply_text)
            if embedding is None:
                embedding = [0.0] * 1024
            reply["embedding"] = embedding
            time.sleep(0.05)

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def embed_directory(self, input_dir: str, output_dir: str):
        """
        对整个目录的文件进行向量化

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
        """
        print("=" * 60)
        print("Comment Network Embedding")
        print("=" * 60)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"API: {self.api_url}")
        print(f"Model: {self.model_name}")
        print("=" * 60)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有 JSON 文件
        input_path = Path(input_dir)
        json_files = sorted(list(input_path.glob("network_news_*.json")))
        
        print(f"\nFound {len(json_files)} files to process")

        # 处理每个文件
        for json_file in tqdm(json_files, desc="Processing files"):
            try:
                output_file = Path(output_dir) / json_file.name
                
                # 跳过已处理的文件 (不打印信息)
                if output_file.exists():
                    continue
                
                self.process_single_file(str(json_file), str(output_file))
                
            except Exception as e:
                print(f"\n  [ERROR] Failed to process {json_file.name}: {e}")
                # 继续处理下一个文件,不中断
                continue

        # 打印统计
        print("\n" + "=" * 60)
        print("Embedding Completed!")
        print("=" * 60)
        print(f"Files processed: {len(json_files)}")
        print(f"API calls - Success: {self.stats['success']}, Failed: {self.stats['failed']}")
        if self.stats['success'] + self.stats['failed'] > 0:
            success_rate = self.stats['success'] / (self.stats['success'] + self.stats['failed']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Embed comment network nodes')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='comment_networks/test',
        help='Input directory containing JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='embeddings/test',
        help='Output directory for embedded JSON files'
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

    args = parser.parse_args()

    # 创建向量化器
    embedder = HeteroGraphEmbedder(
        api_url=args.api_url,
        model_name=args.model
    )

    # 执行向量化
    embedder.embed_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
