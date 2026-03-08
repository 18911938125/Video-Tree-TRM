"""
三层树索引核心数据结构
======================
定义 Video-Tree-TRM 的三层树状索引结构，是所有后续模块
（builder、retriever、losses、pipeline）的基础依赖。

数据结构层次::

    TreeIndex
    └─ List[L1Node]           全局叙事节点
         └─ List[L2Node]      片段级语义节点
              └─ List[L3Node] 帧/细节级节点

与参考项目 (Tree-TRM/video_pyramid.py) 的关键区别：
  - 统一嵌入空间：所有 embedding 均来自 text_embed()，无跨模态问题
  - 序列化方式：pickle 整体序列化（而非 JSON + NPY 分文件存储）
  - L3 全文本化：无需 VisualProjectionLayer
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from utils.logger_system import log_msg


# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------


@dataclass
class IndexMeta:
    """树索引元数据。

    Attributes:
        source_path: 原始数据路径（视频文件或文本文件）。
        modality: 数据模态，"text" 或 "video"。
        embed_model: 嵌入模型名称。
        embed_dim: 嵌入向量维度。
        created_at: 创建时间（ISO 格式字符串）。
    """

    source_path: str
    modality: str
    embed_model: str
    embed_dim: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# 节点数据结构
# ---------------------------------------------------------------------------


@dataclass
class L3Node:
    """L3 帧/细节级节点（叶子层）。

    代表最细粒度的语义单元，对应一个具体的描述片段。

    Attributes:
        id: 节点唯一标识。
        description: 文本描述。
        embedding: 文本嵌入向量，形状 [D]，float32。
        raw_content: 原始文本内容（可选）。
        frame_path: 关联的帧图像路径（可选，仅视频模态）。
        timestamp: 对应的时间戳（秒，可选）。
    """

    id: str
    description: str
    embedding: np.ndarray  # [D]
    raw_content: Optional[str] = None
    frame_path: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass
class L2Node:
    """L2 片段级语义节点（中间层）。

    连接 L1 宏观叙事与 L3 细节描述。

    Attributes:
        id: 节点唯一标识。
        description: 片段文本描述。
        embedding: 文本嵌入向量，形状 [D]，float32。
        time_range: 时间范围 (start, end)（秒，可选）。
        children: 所属的 L3 子节点列表。
    """

    id: str
    description: str
    embedding: np.ndarray  # [D]
    time_range: Optional[Tuple[float, float]] = None
    children: List[L3Node] = field(default_factory=list)


@dataclass
class L1Node:
    """L1 全局叙事节点（根层）。

    代表最粗粒度的语义单元，包含宏观事件摘要。

    Attributes:
        id: 节点唯一标识。
        summary: 高层叙事摘要。
        embedding: 文本嵌入向量，形状 [D]，float32。
        time_range: 时间范围 (start, end)（秒，可选）。
        children: 所属的 L2 子节点列表。
    """

    id: str
    summary: str
    embedding: np.ndarray  # [D]
    time_range: Optional[Tuple[float, float]] = None
    children: List[L2Node] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 树索引容器
# ---------------------------------------------------------------------------


@dataclass
class TreeIndex:
    """三层树索引容器。

    组织和管理三层节点结构，提供嵌入矩阵提取、节点访问、
    以及 pickle 序列化/反序列化接口。

    典型工作流::

        # 1. 构建索引
        index = TreeIndex(metadata=meta, roots=[l1_node_1, l1_node_2])

        # 2. 提取嵌入矩阵（用于 Tree-TRM 检索）
        M_L1 = index.l1_embeddings()                # [N1, D]
        M_L2 = index.l2_embeddings_of(l1_idx=0)     # [N2, D]
        M_L3 = index.l3_embeddings_of(0, 1)         # [N3, D]

        # 3. 序列化
        index.save("cache/my_index.pkl")
        loaded = TreeIndex.load("cache/my_index.pkl")

    Attributes:
        metadata: 索引元数据。
        roots: L1 节点列表。
    """

    metadata: IndexMeta
    roots: List[L1Node] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # 嵌入矩阵提取
    # ------------------------------------------------------------------ #

    def l1_embeddings(self) -> np.ndarray:
        """返回所有 L1 节点的嵌入矩阵。

        返回:
            形状 [N1, D] 的 float32 矩阵。空树返回 [0, D]。
        """
        if not self.roots:
            return np.zeros((0, self.metadata.embed_dim), dtype=np.float32)
        return np.stack([r.embedding for r in self.roots], axis=0).astype(np.float32)

    def l2_embeddings_of(self, l1_idx: int) -> np.ndarray:
        """返回指定 L1 节点下所有 L2 子节点的嵌入矩阵。

        参数:
            l1_idx: L1 节点索引。

        返回:
            形状 [N2, D] 的 float32 矩阵。

        异常:
            IndexError: l1_idx 越界。
        """
        if not (0 <= l1_idx < len(self.roots)):
            raise IndexError(f"l1_idx={l1_idx} 越界，L1 节点数={len(self.roots)}")
        children = self.roots[l1_idx].children
        if not children:
            return np.zeros((0, self.metadata.embed_dim), dtype=np.float32)
        return np.stack([c.embedding for c in children], axis=0).astype(np.float32)

    def l3_embeddings_of(self, l1_idx: int, l2_idx: int) -> np.ndarray:
        """返回指定 L2 节点下所有 L3 子节点的嵌入矩阵。

        参数:
            l1_idx: L1 节点索引。
            l2_idx: L2 节点索引（相对于 L1）。

        返回:
            形状 [N3, D] 的 float32 矩阵。

        异常:
            IndexError: 索引越界。
        """
        if not (0 <= l1_idx < len(self.roots)):
            raise IndexError(f"l1_idx={l1_idx} 越界，L1 节点数={len(self.roots)}")
        l2_children = self.roots[l1_idx].children
        if not (0 <= l2_idx < len(l2_children)):
            raise IndexError(f"l2_idx={l2_idx} 越界，L2 节点数={len(l2_children)}")
        l3_children = l2_children[l2_idx].children
        if not l3_children:
            return np.zeros((0, self.metadata.embed_dim), dtype=np.float32)
        return np.stack([c.embedding for c in l3_children], axis=0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 节点访问
    # ------------------------------------------------------------------ #

    def get_node(self, l1: int, l2: int, l3: int) -> L3Node:
        """按三级路径索引获取 L3 节点。

        参数:
            l1: L1 节点索引。
            l2: L2 节点索引。
            l3: L3 节点索引。

        返回:
            目标 L3Node。

        异常:
            IndexError: 任意层级索引越界。
        """
        if l1 < 0 or l1 >= len(self.roots):
            raise IndexError(f"l1={l1} 越界，L1 节点数={len(self.roots)}")
        l2_children = self.roots[l1].children
        if l2 < 0 or l2 >= len(l2_children):
            raise IndexError(f"l2={l2} 越界，L2 节点数={len(l2_children)}")
        l3_children = l2_children[l2].children
        if l3 < 0 or l3 >= len(l3_children):
            raise IndexError(f"l3={l3} 越界，L3 节点数={len(l3_children)}")
        return l3_children[l3]

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """将整个树索引序列化到磁盘（pickle 格式）。

        参数:
            path: 保存文件路径（推荐 .pkl 后缀）。
        """
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log_msg("INFO", f"树索引已保存至 {path}", n_l1=len(self.roots))

    @classmethod
    def load(cls, path: str) -> "TreeIndex":
        """从磁盘加载树索引。

        .. warning::
            pickle 反序列化可执行任意代码，切勿加载不受信任的文件。
            如需安全替代方案，请考虑 JSON + NPY 分文件存储。

        参数:
            path: pickle 文件路径。

        返回:
            TreeIndex 实例。

        异常:
            FileNotFoundError: 文件不存在。
            TypeError: 文件内容不是 TreeIndex 实例。
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            msg = f"文件内容不是 TreeIndex 实例: {type(obj)}"
            log_msg("ERROR", msg, path=path)
            raise TypeError(msg)
        log_msg("INFO", f"树索引已从 {path} 加载", n_l1=len(obj.roots))
        return obj
