import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import networkx as nx

import src.config.logging_config as logging_config
from src.knowledgeGraph.graph_extractor import (
    extract_concepts_from_text as get_concepts,
)
from src.knowledgeGraph.graph_extractor import (
    extract_relations_from_text as get_relations,
)
from src.util.getHashValue import get_hash_value as hash

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeDocument:
    """文档结构"""

    title: str
    content: str

    def to_dict(self) -> dict:
        """转换为字典（用于存储）"""
        return asdict(self)


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


def build_concept_graph(documents: list[KnowledgeDocument]) -> nx.DiGraph:
    """
    从文档列表构建概念图
    文档：node(uuid,type="document",title="文档标题，这里等于文件名")
    概念：node(concept_name,type="concept")
    文档 - 概念：edge(uuid,concept_name,relation="mentions")
    概念 - 概念：edge(concept_name,concept_name,relation=relation_from_llm")
    Args:
        document:文档列表，每个文档包含 {'title':str,'content':str}

    Returns:
        NetworkX 图对象
    """
    G = nx.DiGraph()

    for doc in documents:
        title = doc.title
        content = doc.content
        if len(title) == 0 or len(content) == 0:
            continue
        fulll_text = f"<文本标题>{title}</文本标题>\n<文本内容>{content}</文本内容>"
        title_hash_uuid = hash(title)
        concepts = get_concepts(fulll_text)

        if not concepts:
            continue
        logger.info(f"title:{title},concepts :{concepts}")
        G.add_node(title_hash_uuid, type=NodeType.DOCUMENT.value, title=title)
        for concept in concepts:
            G.add_node(concept, type=NodeType.CONCEPT.value)
            G.add_edge(title_hash_uuid, concept, relation=RelationType.MENTION.value)

        relations = get_relations(fulll_text, concepts)
        for source, realtion, target in relations:
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, relation=realtion)

    return G


def search_by_node(G: nx.DiGraph, concept: str, depth: int = 2) -> list[dict[str, Any]]:
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


# documents = [
#     KnowledgeDocument(
#         title="日本负利率政策",
#         content="负利率是一种货币政策，由央行实施，影响银行利润和信贷政策",
#     ),
#     KnowledgeDocument(
#         title="量化宽松",
#         content="量化宽松是一种货币政策，包括购买国债，影响央行资产负债表",
#     ),
#     KnowledgeDocument(
#         title="通胀目标",
#         content="通胀目标是货币政策的目标之一，央行通过调整利率实现通胀目标",
#     ),
# ]
# logger.info(documents)

# graph = build_concept_graph(documents=documents)

# logger.info(search_by_concept(graph, "通胀目标", 2))
# logger.info(get_graph_stats(G=graph))


# visualize_graph(graph, "logs/", "概念图_测试01")
