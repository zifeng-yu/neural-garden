import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import networkx as nx

import src.config.logging_config as logging_config
from src.repository.concept_relations import query_all as query_all_concept_relations
from src.repository.document_chunk_concepts import query_all as query_all_chunk_concepts
from src.repository.documents import query_all as query_all_documents

logger = logging.getLogger(__name__)


class NodeAttribute(str, Enum):
    TYPE = "type"
    # 只有 type = document 才存在
    TITLE = "title"


class NodeType(str, Enum):
    """
    node type enum, key name = type
    """

    # 文档
    DOCUMENT = "document"
    # 概念
    CONCEPT = "concept"


class RelationType(str, Enum):
    """
    edge type enum, key name = relation
    目前依赖llm返回 ，在prompt给例子
    """

    # doc -> concept 的关系
    MENTION = "mentions"


def build_concept_graph() -> nx.MultiDiGraph:
    """
    新版 只需要从sqlite读取所有数据，即可内存建立图
    """
    # 因为现在按照chunk产出概念，再归一，所以可能存在多边（概念A -> 概念B）
    G = nx.MultiDiGraph()
    # 1. 拉取数据
    all_chunk_concetps = query_all_chunk_concepts()
    all_documents = query_all_documents()
    all_concept_relations = query_all_concept_relations()
    # 2. 生成图谱
    for docment in all_documents:
        G.add_node(docment.id, type=NodeType.DOCUMENT.value, title=docment.file_name)
    for chunk_concept in all_chunk_concetps:
        G.add_node(chunk_concept.normalized_concept, type=NodeType.CONCEPT.value)
        G.add_edge(
            chunk_concept.document_id,
            chunk_concept.normalized_concept,
            relation=RelationType.MENTION.value,
        )

    for concept_relation in all_concept_relations:
        G.add_edge(
            concept_relation.source_concept,
            concept_relation.target_concept,
            relation=concept_relation.relation,
        )

    return G


def search_by_node(
    G: nx.MultiDiGraph, concept: str, depth: int = 2
) -> list[dict[str, Any]]:
    """
    按概念搜索关联概念
    Args:
        G: NetworkX 图对象
        concept: 起始概念
        depth: 搜索深度(默认2层)
    Returns:
        关联概念列表
    """
    if concept not in G.nodes:
        logger.info(f"⚠️  概念 '{concept}' 不在图中")
        return []

    related_concepts = []
    for node, shortest_path_length in nx.single_source_shortest_path_length(
        G, concept, cutoff=depth
    ).items():
        if node != concept:
            result = {
                "id": node,
                "shortest_path_length": shortest_path_length,
                **G.nodes[node],
            }
            related_concepts.append(result)

    return related_concepts
