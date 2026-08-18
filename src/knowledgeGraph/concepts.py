import logging
from collections import defaultdict

import networkx as nx

import src.config.logging_config as logging_config
from src.embedding.getEmbedding import get_embedding
from src.knowledgeGraph.concepts_extractor import (
    extract_concepts_from_text,
    extract_max_similarity_concept,
)
from src.knowledgeGraph.concepts_relation import (
    ConceptClusters,
    ConceptInChunkText,
    ConceptMergeResult,
    extract_concept_clusters,
)
from src.repository.document_chunk_concepts import (
    query_by_normalized_concept,
)
from src.repository.document_chunks import query_by_ids as query_by_ids_chunks
from src.repository.document_insert_domain import InsertChunk
from src.similarity import calculate_similarity
from src.util.getHashValue import get_hash_value as hash
from src.vector_store.query_dao import search_by_threshold_concept

logger = logging.getLogger(__name__)


def documents_to_concepts(knowledgeUnit_title: str, chunk_content: str):
    """
    提炼所有文档的概念
    """
    result = []
    if knowledgeUnit_title is None or chunk_content is None:
        return result
    if len(knowledgeUnit_title) == 0 or len(chunk_content) == 0:
        return result
    fulll_text = f"<文本标题>{knowledgeUnit_title}</文本标题>\n<文本内容>{chunk_content}</文本内容>"
    return extract_concepts_from_text(fulll_text)


def normalized_concept(
    insert_chunks: list[InsertChunk],
):
    """概念文档外部归一化 使用向量检索,llm综合判断
    args: 新概念，chunk
    """
    if not insert_chunks:
        return []
    # 0. assembler
    concept_chunk_dict = defaultdict(list[str])
    for chunk in insert_chunks:
        for concept_x in chunk.concepts:
            concept_chunk_dict[concept_x.normalized_concept].append(chunk.content)
    concepts = [x for x in concept_chunk_dict]
    # 1. 归一
    concept_normalized_dict = {}
    for concept in concepts:
        # 2. 向量相似度查询
        concept_results_similarity = search_by_threshold_concept(query=concept)
        normalized_concept_result = concept
        if concept_results_similarity:
            # sqlite查询
            similarity_concepts = []
            for x in concept_results_similarity:
                # 3. llm最后判断
                query_by_normalized_concept_result = query_by_normalized_concept(
                    x.normalized_concept
                )
                if query_by_normalized_concept_result:
                    chunks = query_by_ids_chunks(
                        [
                            conceptDO.document_chunk_id
                            for conceptDO in query_by_normalized_concept_result
                        ]
                    )
                    similarity_concepts.append(
                        (x.normalized_concept, [chunk.content for chunk in chunks])
                    )
            max_similarity_concepts = extract_max_similarity_concept(
                concept=(concept, concept_chunk_dict[concept]),
                similarity_concepts=similarity_concepts,
            )
            if max_similarity_concepts is not None and len(max_similarity_concepts) > 0:
                logger.info(
                    f"新增加概念 {concept} 最相似概念 llm判断结果：{max_similarity_concepts}"
                )
                normalized_concept_result = max_similarity_concepts[0]
        concept_normalized_dict[concept] = normalized_concept_result
    # 2. 改变对象
    for insert in insert_chunks:
        for con in insert.concepts:
            con.normalized_concept = concept_normalized_dict.get(
                con.normalized_concept, con.normalized_concept
            )
            con.normalized_concept_hash = hash(con.normalized_concept)


def _find_similar_concept_pairs(
    concepts: list[str],
    embeddings: list[list[float]],
    similarity_threshold: float = 0.2,
) -> list[tuple[str, str, float]]:
    calculate_similarity_result = calculate_similarity(embeddings)
    result_pairs = []
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            result_similarity = calculate_similarity_result[i][j]
            if result_similarity > similarity_threshold:
                result_pairs.append(
                    (concepts[i], concepts[j], float(similarity_threshold))
                )
    return result_pairs


def _concept_paris_connected_components(
    concept_pairs: list[tuple[str, str, float]],
) -> list[set[str]]:
    if not concept_pairs:
        return []
    G = nx.Graph()
    for a, b, score in concept_pairs:
        G.add_edge(a, b, weight=score)
    return list(nx.connected_components(G))


def concept_clustering_in_document(insert_chunks: list[InsertChunk]):
    """文档内部概念归一"""
    if not insert_chunks:
        return
    # 0. assembler
    concept_chunk_dict = defaultdict(list)
    for chunk in insert_chunks:
        for concept_x in chunk.concepts:
            concept_chunk_dict[concept_x.normalized_concept].append(chunk.content)
    concepts = [x for x in concept_chunk_dict]
    # 1. embedding
    concepts_embedding = [get_embedding(e) for e in concepts]
    # 2. 相似度pairs 有可能阈值太高 返回为空
    concept_pairs = _find_similar_concept_pairs(concepts, concepts_embedding)
    logger.info(f"concept_pairs result {concept_pairs}")
    if not concept_pairs:
        # 为空
        return
    # 3. graph找寻连通分量
    concepts_connnected = _concept_paris_connected_components(concept_pairs)
    logger.info(f"concepts_connnected result {concepts_connnected}")
    # 4. llm归一(聚集)
    concept_normalized_dict = {}
    for concept_nodes in concepts_connnected:
        conceptInChunkTexts: list[ConceptInChunkText] = []
        for node in concept_nodes:
            conceptInChunkTexts.append(
                ConceptInChunkText(node, concept_chunk_dict[node])
            )
        result: ConceptMergeResult = extract_concept_clusters(conceptInChunkTexts)
        logger.info(f"extract_concept_clusters result {result}")
        for x in result.clusters:
            concept_normalized_dict[x.canonical] = x.canonical
            for y in x.members:
                concept_normalized_dict[y] = x.canonical
        for un in result.unmerged:
            concept_normalized_dict[un] = un
    # 5. 改变对象
    for insert in insert_chunks:
        for con in insert.concepts:
            con.normalized_concept = concept_normalized_dict.get(
                con.normalized_concept, con.normalized_concept
            )
            con.normalized_concept_hash = hash(con.normalized_concept)
