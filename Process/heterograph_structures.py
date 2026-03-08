"""
异构图数据结构定义
用于构建包含作者、新闻、评论三类节点的异构图
"""
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import numpy as np
import json


class NodeType(str, Enum):
    """节点类型枚举"""
    AUTHOR = "author"
    NEWS = "news"
    COMMENT = "comment"


class AuthorNode(BaseModel):
    """作者节点"""
    node_id: str = Field(..., description="节点唯一标识")
    node_type: NodeType = Field(default=NodeType.AUTHOR, description="节点类型")
    author_id: str = Field(default="", description="作者ID（原始数据中的ID）")
    name: str = Field(default="", description="作者名称")
    features: Optional[List[float]] = Field(default=None, description="节点特征向量（一维列表形式）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他元数据")

    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist()
        }

    @validator('features')
    def validate_features(cls, v):
        if v is not None and not isinstance(v, list):
            raise ValueError('features must be a list')
        return v

    def get_features_array(self) -> Optional[np.ndarray]:
        """获取 numpy 数组形式的特征"""
        if self.features:
            return np.array(self.features)
        return None


class NewsNode(BaseModel):
    """新闻节点"""
    node_id: str = Field(..., description="节点唯一标识")
    node_type: NodeType = Field(default=NodeType.NEWS, description="节点类型")
    news_id: str = Field(default="", description="新闻ID")
    author_id: str = Field(default="", description="发布该新闻的作者ID")
    text: str = Field(default="", description="新闻文本内容")
    publish_time: str = Field(default="", description="发布时间")
    features: Optional[List[float]] = Field(default=None, description="节点特征向量")
    label: Optional[int] = Field(default=None, description="标签（谣言/非谣言等）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他元数据")

    @validator('label')
    def validate_label(cls, v):
        if v is not None and v < 0:
            raise ValueError('label must be non-negative')
        return v

    def get_features_array(self) -> Optional[np.ndarray]:
        """获取 numpy 数组形式的特征"""
        if self.features:
            return np.array(self.features)
        return None


class CommentNode(BaseModel):
    """评论节点"""
    node_id: str = Field(..., description="节点唯一标识")
    node_type: NodeType = Field(default=NodeType.COMMENT, description="节点类型")
    comment_id: str = Field(default="", description="评论ID")
    news_id: str = Field(default="", description="所属新闻ID")
    author_id: str = Field(default="", description="评论作者ID")
    text: str = Field(default="", description="评论文本内容")
    publish_time: str = Field(default="", description="发布时间")
    parent_comment_id: Optional[str] = Field(default=None, description="父评论ID")
    features: Optional[List[float]] = Field(default=None, description="节点特征向量")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他元数据")

    def get_features_array(self) -> Optional[np.ndarray]:
        """获取 numpy 数组形式的特征"""
        if self.features:
            return np.array(self.features)
        return None


class Edge(BaseModel):
    """有向边"""
    source_id: str = Field(..., description="源节点ID")
    target_id: str = Field(..., description="目标节点ID")
    weight: float = Field(default=1.0, ge=0, description="边的权重")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他元数据")

    @validator('source_id', 'target_id')
    def validate_node_ids(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('node_id must be a non-empty string')
        return v


class HeteroGraph(BaseModel):
    """异构图结构"""
    graph_id: str = Field(..., description="图的唯一标识（例如：news_id）")

    # 节点存储
    author_nodes: Dict[str, AuthorNode] = Field(default_factory=dict, description="作者节点字典")
    news_nodes: Dict[str, NewsNode] = Field(default_factory=dict, description="新闻节点字典")
    comment_nodes: Dict[str, CommentNode] = Field(default_factory=dict, description="评论节点字典")

    # 所有边
    edges: List[Edge] = Field(default_factory=list, description="所有边的列表")

    # 邻接表索引
    adjacency_out: Dict[str, List[str]] = Field(default_factory=dict, description="出边邻接表")
    adjacency_in: Dict[str, List[str]] = Field(default_factory=dict, description="入边邻接表")

    # 元路径定义
    metapaths: List[List[str]] = Field(default_factory=list, description="元路径列表")

    class Config:
        arbitrary_types_allowed = True

    def add_node(self, node):
        """添加节点"""
        if isinstance(node, AuthorNode):
            self.author_nodes[node.node_id] = node
        elif isinstance(node, NewsNode):
            self.news_nodes[node.node_id] = node
        elif isinstance(node, CommentNode):
            self.comment_nodes[node.node_id] = node
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def add_edge(self, edge: Edge):
        """添加有向边"""
        self.edges.append(edge)

        # 更新邻接表
        if edge.source_id not in self.adjacency_out:
            self.adjacency_out[edge.source_id] = []
        self.adjacency_out[edge.source_id].append(edge.target_id)

        if edge.target_id not in self.adjacency_in:
            self.adjacency_in[edge.target_id] = []
        self.adjacency_in[edge.target_id].append(edge.source_id)

    def get_node(self, node_id: str):
        """根据ID获取节点"""
        if node_id in self.author_nodes:
            return self.author_nodes[node_id]
        elif node_id in self.news_nodes:
            return self.news_nodes[node_id]
        elif node_id in self.comment_nodes:
            return self.comment_nodes[node_id]
        return None

    def get_node_type(self, node_id: str) -> Optional[NodeType]:
        """获取节点类型"""
        node = self.get_node(node_id)
        return node.node_type if node else None

    def get_edges_by_type(self, source_type: NodeType, target_type: NodeType) -> List[Edge]:
        """根据源节点和目标节点类型获取边"""
        return [
            edge for edge in self.edges
            if (self.get_node_type(edge.source_id) == source_type and
                self.get_node_type(edge.target_id) == target_type)
        ]

    def get_edge_index(self) -> Tuple[Tuple[List[int], List[int]], Dict[str, int]]:
        """
        获取边的索引格式（用于PyG等图神经网络库）
        返回：([source_idx_list], [target_idx_list]), node_id_to_idx
        """
        node_id_to_idx = self._build_node_index_mapping()

        source_indices = []
        target_indices = []

        for edge in self.edges:
            src_idx = node_id_to_idx.get(edge.source_id, -1)
            tgt_idx = node_id_to_idx.get(edge.target_id, -1)
            if src_idx >= 0 and tgt_idx >= 0:
                source_indices.append(src_idx)
                target_indices.append(tgt_idx)

        return (source_indices, target_indices), node_id_to_idx

    def _build_node_index_mapping(self) -> Dict[str, int]:
        """构建节点ID到索引的映射"""
        mapping = {}
        idx = 0

        for node_id in self.author_nodes:
            mapping[node_id] = idx
            idx += 1
        for node_id in self.news_nodes:
            mapping[node_id] = idx
            idx += 1
        for node_id in self.comment_nodes:
            mapping[node_id] = idx
            idx += 1

        return mapping

    def get_node_types_mask(self) -> Dict[str, List[bool]]:
        """获取节点类型的mask"""
        node_id_to_idx = self._build_node_index_mapping()
        total_nodes = len(node_id_to_idx)

        masks = {
            NodeType.AUTHOR: [False] * total_nodes,
            NodeType.NEWS: [False] * total_nodes,
            NodeType.COMMENT: [False] * total_nodes
        }

        for node_id, idx in node_id_to_idx.items():
            node_type = self.get_node_type(node_id)
            masks[node_type][idx] = True

        return masks

    def define_metapaths(self):
        """定义元路径"""
        self.metapaths = [
            # 1. 新闻 - 作者 - 新闻（捕获作者的写作风格、立场）
            [NodeType.NEWS, NodeType.AUTHOR, NodeType.NEWS],
            # 2. 新闻 - 评论 - 新闻（传播效果、话题延续）
            [NodeType.NEWS, NodeType.COMMENT, NodeType.NEWS],
            # 3. 新闻 - 评论 - 作者（读者对作者的反馈）
            [NodeType.NEWS, NodeType.COMMENT, NodeType.AUTHOR],
            # 4. 评论 - 作者 - 评论（作者-读者互动）
            [NodeType.COMMENT, NodeType.AUTHOR, NodeType.COMMENT],
            # 5. 评论 - 评论 - 评论（读者间互动）
            [NodeType.COMMENT, NodeType.COMMENT, NodeType.COMMENT],
        ]

    def save_to_json(self, filepath: str):
        """保存为JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str):
        """从JSON文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def get_statistics(self) -> Dict[str, Any]:
        """获取图的统计信息"""
        return {
            "graph_id": self.graph_id,
            "num_author_nodes": len(self.author_nodes),
            "num_news_nodes": len(self.news_nodes),
            "num_comment_nodes": len(self.comment_nodes),
            "total_nodes": len(self.author_nodes) + len(self.news_nodes) + len(self.comment_nodes),
            "num_edges": len(self.edges),
            "num_metapaths": len(self.metapaths) if self.metapaths else 0
        }
