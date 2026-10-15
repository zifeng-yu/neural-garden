import re
from itertools import zip_longest

import networkx as nx

from src.config.config import CHROMA_CONCEPT_TABLE_NAME
from src.embedding.getEmbedding import get_embedding
from src.get_chroma_collection import get_existing_collection
from src.knowledgeGraph.knowledge_graph import RelationType
from src.repository.concept_relations import query_by_id as query_relation_id_concept
from src.repository.document_chunk_concepts import query_all as query_concepts_all
from src.repository.document_chunk_concepts import (
    query_by_chunk_id_document_id as concepts_chunk_document_query,
)
from src.repository.document_chunk_concepts import query_by_normalized_concept
from src.repository.document_chunk_knowledge_units import (
    query_by_document_id_chunk_id as knowledge_document_chunk_id,
)
from src.repository.relation_evidence import (
    query_by_relation_id_evidence_role_both,
)
from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum
from src.repository.search_sessions import save_search_session_with_results
from src.retrieval.hybrid_search import fuse_rrf
from src.retrieval.search_result import SearchResult


def _match_concepts(
    query: str,
    *,
    seed_limit: int = 3,
    vector_candidate_k: int = 10,
    min_similarity: float = 0.5,
) -> list[str]:
    """名称匹配优先；没有名称命中时，再进行概念向量匹配。"""
    if query is None or len(query.strip()) == 0:
        return []
    # 名称匹配
    concepts = query_concepts_all()
    if not concepts:
        return []
    name_pairs = set()
    for concept in concepts:
        name_pairs.add((concept.concept, concept.normalized_concept))
        name_pairs.add((concept.normalized_concept, concept.normalized_concept))

    matches = []
    query_text = query.casefold()
    for name, normalized in sorted(name_pairs):
        if not name:
            continue
        name_text = name.casefold()
        # 英文端点不能紧邻字母、数字或下划线，避免 AI 命中 explain。
        # 不用 Unicode 的 \b：中文句子中的“介绍AI技术”仍应命中 AI，
        # 纯中文概念保留子串匹配；转义后 C++ 等名称中的符号按字面匹配。
        pattern = re.escape(name_text)
        if name_text[0].isascii():
            pattern = r"(?<![a-z0-9_])" + pattern
        if name_text[-1].isascii():
            pattern += r"(?![a-z0-9_])"
        if re.search(pattern, query_text):
            matches.append(normalized)

    if matches:
        return list(dict.fromkeys(matches))[:seed_limit]

    # 名称匹配失败，开始向量
    concept_collection = get_existing_collection(CHROMA_CONCEPT_TABLE_NAME)
    if concept_collection.count() == 0:
        return []

    query_vector = get_embedding(query)
    if query_vector is None:
        raise RuntimeError("query embedding error")

    results = concept_collection.query(
        query_embeddings=query_vector,
        n_results=min(vector_candidate_k, concept_collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    concept_results = []

    documents = results.get("documents")
    distances = results.get("distances")

    if not documents or not distances:
        return []
    valid_names = {c.normalized_concept for c in concepts}
    for doc, distance in zip(documents[0], distances[0], strict=True):
        if doc is None:
            continue
        similarity = 1 - distance
        if doc in valid_names and similarity >= min_similarity:
            concept_results.append(doc)

    return list(dict.fromkeys(concept_results))[:seed_limit]


def _concepts_from_hits(
    hits: list[SearchResult],
    *,
    seed_limit: int = 3,
) -> list[str]:
    """查询基础候选已经关联的概念，返回标准概念名称。"""
    if not hits:
        return []
    if type(seed_limit) is not int or seed_limit <= 0:
        raise ValueError("seed_limit 必须是正整数")
    seeds = []
    for hit in sorted(hits, key=lambda item: item.rank):
        if hit.document_id is None or hit.document_chunk_id is None:
            continue
        concepts = concepts_chunk_document_query(hit.document_chunk_id, hit.document_id)
        for concept in sorted(
            concepts,
            key=lambda item: item.normalized_concept,
        ):
            if concept.normalized_concept not in seeds:
                seeds.append(concept.normalized_concept)

            if len(seeds) >= seed_limit:
                return seeds

    return seeds


def _expand_from_concepts(
    graph: nx.MultiDiGraph,
    seed_concepts: list[str],
    *,
    candidate_k: int = 10,
    relation_limit_per_seed: int = 4,
) -> list[SearchResult]:
    """查起点及一跳邻居关联的知识单元，去重、排序后返回候选。"""
    if not seed_concepts:
        return []
    if type(relation_limit_per_seed) is not int or relation_limit_per_seed <= 0:
        raise ValueError("relation_limit_per_seed 必须为整数且大于0")
    # 查起点关联的知识单元
    hits_by_seed = []
    for seed in seed_concepts:
        concept_dos = query_by_normalized_concept(seed)
        if not concept_dos:
            continue
        for concept_do in concept_dos:
            knowledge_unit_dos = knowledge_document_chunk_id(
                concept_do.document_id, concept_do.document_chunk_id
            )
            if not knowledge_unit_dos:
                continue
            for knowledge_unit_do in knowledge_unit_dos:
                hits_by_seed.append(
                    {
                        "content": knowledge_unit_do.embedding_text,
                        "document_id": knowledge_unit_do.document_id,
                        "document_chunk_id": knowledge_unit_do.document_chunk_id,
                    }
                )

    if not graph:
        return _assember_searchResult_list(hits_by_seed, None, candidate_k)
    # 查一跳 关联数据
    hits_by_graph = []
    for node in seed_concepts:
        if not graph.has_node(node):
            continue
        seen_relation_id: set[int] = set()

        out_edges_list = _prepare_edges(graph.out_edges(node, keys=True, data=True))
        in_edges_list = _prepare_edges(graph.in_edges(node, keys=True, data=True))
        for out_edge, in_edge in zip_longest(out_edges_list, in_edges_list):
            if out_edge is not None:
                source, target, edge_key, edge_data = out_edge
                hit_by_graph = _assember_graph_edges(
                    target, edge_data, seen_relation_id
                )
                if hit_by_graph is not None:
                    hits_by_graph.extend(hit_by_graph)

            if len(seen_relation_id) >= relation_limit_per_seed:
                break

            if in_edge is not None:
                source, target, edge_key, edge_data = in_edge
                hit_by_graph = _assember_graph_edges(
                    source, edge_data, seen_relation_id
                )
                if hit_by_graph is not None:
                    hits_by_graph.extend(hit_by_graph)

            if len(seen_relation_id) >= relation_limit_per_seed:
                break

    return _assember_searchResult_list(hits_by_seed, hits_by_graph, candidate_k)


def _prepare_edges(edges):
    return sorted(
        (
            edge
            for edge in edges
            if edge[3].get("relation_id") is not None
            and edge[3].get("relation") is not None
            and edge[3].get("relation") != "mentions"
        ),
        key=lambda edge: edge[3]["relation_id"],
    )


def _assember_graph_edges(
    edge_node, edge_data, seen_relation_id: set
) -> list[dict] | None:
    relation_id = edge_data.get(RelationType.RELATION_ID_KEY_NAME.value)
    if relation_id in seen_relation_id:
        return None

    concept_relation = query_relation_id_concept(relation_id)
    if concept_relation is None:
        return None
    evidence = query_by_relation_id_evidence_role_both(relation_id)
    if evidence is None:
        return None
    chunks_concept = query_by_normalized_concept(edge_node)
    if not chunks_concept:
        return None
    seen_relation_id.add(relation_id)
    return_list = []
    for chunk_concept in chunks_concept:
        knowledge_unit_dos = knowledge_document_chunk_id(
            chunk_concept.document_id, chunk_concept.document_chunk_id
        )
        if not knowledge_unit_dos:
            continue
        for knowledge_unit_do in knowledge_unit_dos:
            return_list.append(
                {
                    "content": knowledge_unit_do.embedding_text,
                    "document_id": knowledge_unit_do.document_id,
                    "document_chunk_id": knowledge_unit_do.document_chunk_id,
                }
            )
    return return_list


def _assember_searchResult_list(
    hits_by_seed, hits_by_graph, candidate_k
) -> list[SearchResult]:
    if hits_by_seed is None:
        hits_by_seed = []
    if hits_by_graph is None:
        hits_by_graph = []

    unique_hits_by_seed = []
    seen_hits_by_seed = set()
    for hit_seed in hits_by_seed:
        key = (hit_seed["document_id"], hit_seed["document_chunk_id"])
        if key in seen_hits_by_seed:
            continue

        seen_hits_by_seed.add(key)
        unique_hits_by_seed.append(hit_seed)

    unique_hits_by_graph = []
    seen_hits_by_graph = set()
    for hit_graph in hits_by_graph:
        key = (hit_graph["document_id"], hit_graph["document_chunk_id"])
        if key in seen_hits_by_graph:
            continue

        seen_hits_by_graph.add(key)
        unique_hits_by_graph.append(hit_graph)

    # unique_hits_by_seed unique_hits_by_graph 去重 融合 排序
    unique_hits_by_seed.sort(
        key=lambda hit: (hit["document_id"], hit["document_chunk_id"])
    )
    unique_hits_by_graph = [
        hit
        for hit in unique_hits_by_graph
        if (hit["document_id"], hit["document_chunk_id"]) not in seen_hits_by_seed
    ]
    unique_hits_by_graph.sort(
        key=lambda hit: (hit["document_id"], hit["document_chunk_id"])
    )

    # 融合
    hits = []
    for seed_hit, graph_hit in zip_longest(unique_hits_by_seed, unique_hits_by_graph):
        if seed_hit is not None:
            hits.append(seed_hit)
        if graph_hit is not None:
            hits.append(graph_hit)

    hits = hits[:candidate_k]
    result_list: list[SearchResult] = []
    for rank, hit in enumerate(hits, start=1):
        result_list.append(
            SearchResult(
                hit["content"],
                hit["document_id"],
                hit["document_chunk_id"],
                1 / rank,
                rank,
                SourceTypeEnum.GRAPH_KNOWLEDGE,
                RetrievalTypeEnum.GRAPH,
                None,
            )
        )
    return result_list


def graph_search_by_query(
    query: str,
    graph: nx.MultiDiGraph,
    top_k: int,
    *,
    min_similarity: float,
    seed_limit: int = 3,
    vector_candidate_k: int = 10,
    relation_limit_per_seed: int = 4,
    save_session: bool = True,
) -> list[SearchResult]:
    seeds = _match_concepts(
        query,
        seed_limit=seed_limit,
        vector_candidate_k=vector_candidate_k,
        min_similarity=min_similarity,
    )

    result = _expand_from_concepts(
        graph,
        seeds,
        candidate_k=top_k,
        relation_limit_per_seed=relation_limit_per_seed,
    )

    if save_session:
        save_search_session_with_results(
            query, result, RetrievalTypeEnum.GRAPH, SourceTypeEnum.GRAPH_KNOWLEDGE
        )

    return result


def graph_search_by_hits(
    graph: nx.MultiDiGraph,
    hits: list[SearchResult],
    top_k: int,
    *,
    query: str,
    seed_limit: int = 3,
    candidate_k: int = 10,
    relation_limit_per_seed: int = 4,
    save_session: bool = True,
) -> list[SearchResult]:
    seeds = _concepts_from_hits(hits, seed_limit=seed_limit)
    graph_hits = _expand_from_concepts(
        graph,
        seeds,
        candidate_k=candidate_k,
        relation_limit_per_seed=relation_limit_per_seed,
    )
    result = fuse_rrf(hits, graph_hits, top_k=top_k)
    if save_session:
        save_search_session_with_results(
            query, result, RetrievalTypeEnum.HYBRID, SourceTypeEnum.HYBRID_KNOWLEDGE
        )

    return result


if __name__ == "__main__":
    pass
