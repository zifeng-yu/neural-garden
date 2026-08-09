import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import networkx as nx

import src.config.logging_config as logging_config
from src.embedding.getEmbedding import get_embedding
from src.knowledgeGraph.graph_extractor import (
    extract_concepts_from_text as get_concepts,
)
from src.knowledgeGraph.graph_extractor import extract_max_similarity_concept
from src.knowledgeGraph.graph_extractor import (
    extract_relations_from_text as get_relations,
)
from src.util.getHashValue import get_hash_value as hash
from src.vector_store.query_dao import (
    get_by_id_concept,
    search_by_threshold_concept,
)
from src.vector_store.save_dao import ConceptDO, ConceptMetadata, save_concept

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


def get_documents_to_concepts(
    documents: list[KnowledgeDocument],
) -> list[dict[str, Any]]:
    """
    提炼所有文档的概念
    """
    result = []
    for doc in documents:
        title = doc.title
        content = doc.content
        if len(title) == 0 or len(content) == 0:
            continue
        fulll_text = f"<文本标题>{title}</文本标题>\n<文本内容>{content}</文本内容>"
        title_hash_uuid = hash(title)
        concepts = get_concepts(fulll_text)
        result.append(
            {
                "id": title_hash_uuid,
                "title": title,
                "full_text": fulll_text,
                "concepts": concepts,
            }
        )
    return result


def resolve_concept(concept: str) -> dict[str, Any]:
    """概念归一化"""
    concept_results_similarity = search_by_threshold_concept(query=concept)
    if concept_results_similarity is not None and len(concept_results_similarity) > 0:
        max_similarity_concepts = extract_max_similarity_concept(
            concepts=concept,
            similarity_concepts=[x.conceptName for x in concept_results_similarity],
        )
        if max_similarity_concepts is not None and len(max_similarity_concepts) > 0:
            logger.info(
                f"新增加概念 {concept} 最相似概念 llm判断结果：{max_similarity_concepts}"
            )
            return {"is_relove": True, "resolve_concept": max_similarity_concepts[0]}
    return {"is_relove": False}


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
    # 1. 遍历documents得到所有概念 -> list[dict] dict id title full_text concepts
    doc_to_concepts = get_documents_to_concepts(documents)
    # 2. 生成图谱
    for doc_dict in doc_to_concepts:
        if doc_dict["concepts"] is None or len(doc_dict["concepts"]) == 0:
            continue
        G.add_node(
            doc_dict["id"], type=NodeType.DOCUMENT.value, title=doc_dict["title"]
        )
        concepts_ending = []
        for concept in doc_dict["concepts"]:
            # 查询概念是否已经入库
            query_id_chroma = get_by_id_concept(hash(concept))
            before_similarity_concept = concept
            if query_id_chroma is None:
                # 不在库中，
                resolve_concept_result = resolve_concept(concept=concept)
            if resolve_concept_result["is_relove"]:
                before_similarity_concept = resolve_concept_result["resolve_concept"]
            # 查询概念是否已经入库(概念可能更新)
            query_before_similarity_id_chroma = get_by_id_concept(
                hash(before_similarity_concept)
            )
            # 没入库->入库 有入库—>title不在->入库
            if query_before_similarity_id_chroma is None:
                conceptDO = ConceptDO(
                    id=hash(before_similarity_concept),
                    embedding=get_embedding(before_similarity_concept),
                    conceptName=before_similarity_concept,
                    metadata=ConceptMetadata(source=[doc_dict["title"]]),
                )
                save_concept(conceptDO)
            else:
                if (
                    doc_dict["title"]
                    not in query_before_similarity_id_chroma.metadata.source
                ):
                    query_before_similarity_id_chroma.metadata.source.append(
                        doc_dict["title"]
                    )
                save_concept(query_before_similarity_id_chroma)

            G.add_node(before_similarity_concept, type=NodeType.CONCEPT.value)
            G.add_edge(
                doc_dict["id"],
                before_similarity_concept,
                relation=RelationType.MENTION.value,
            )
            concepts_ending.append(before_similarity_concept)

        relations = get_relations(doc_dict["full_text"], concepts_ending)
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
