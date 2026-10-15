"""
Chroma向量检索。
"""

import logging

from src.config.config import CHROMA_KNOWLEDGE_TABLE_NAME
from src.embedding.getEmbedding import get_embedding
from src.get_chroma_collection import get_existing_collection
from src.repository.search_results import (
    RetrievalTypeEnum,
    SourceTypeEnum,
)
from src.repository.search_sessions import save_search_session_with_results
from src.retrieval.search_result import SearchResult

logger = logging.getLogger(__name__)


def search_vector(
    query: str, top_k: int = 3, show_score: bool = True, save_session: bool = True
) -> list[SearchResult]:
    """
    向量搜索入口（增强版）

    Args:
        query: 搜索查询
        top_k: 返回结果数量
        show_score: 是否显示相似度分数
        save_session: 是否在检索成功后合并保存session和结果
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query必须是非空字符串")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k必须是正整数")
    query = query.strip()

    # 生成查询向量并搜索
    embedding = get_embedding(query)
    if embedding is None:
        raise RuntimeError("向量检索失败：未能生成查询 embedding")

    collection_knowledge = get_existing_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
    logger.info(f"knowledge表集合大小 {collection_knowledge.count()}")

    count = collection_knowledge.count()
    if count == 0:
        if save_session:
            save_search_session_with_results(
                query, [], RetrievalTypeEnum.VECTOR, SourceTypeEnum.CHROMA_KNOWLEDGE
            )
        return []

    results_knowledge = collection_knowledge.query(
        query_embeddings=embedding,
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    # 4. 格式化输出（带相似度）
    log_results(results_knowledge, query, show_score=show_score)

    search_result = []
    rank = 1

    if (
        results_knowledge
        and results_knowledge["documents"]
        and results_knowledge["metadatas"]
        and results_knowledge["distances"]
    ):
        for doc, meta, distance in zip(
            results_knowledge["documents"][0],
            results_knowledge["metadatas"][0],
            results_knowledge["distances"][0],
            strict=True,
        ):
            document_id = meta.get("document_id")
            if not isinstance(document_id, int):
                document_id = None
            document_chunk_id = meta.get("document_chunk_id")
            if not isinstance(document_chunk_id, int):
                document_chunk_id = None
            search_result.append(
                SearchResult(
                    doc,
                    document_id,
                    document_chunk_id,
                    1 - distance,
                    rank,
                    SourceTypeEnum.CHROMA_KNOWLEDGE,
                    RetrievalTypeEnum.VECTOR,
                    raw_score=distance,
                )
            )
            rank += 1

    if save_session:
        save_search_session_with_results(
            query,
            search_result,
            RetrievalTypeEnum.VECTOR,
            SourceTypeEnum.CHROMA_KNOWLEDGE,
        )

    return search_result


def log_results(results, query, show_score: bool = True):
    """
    格式化输出搜索结果

    Args:
        results: Chroma 查询结果
        query: 搜索查询
        show_score: 是否显示相似度分数
    """
    logger.info(f"\n🔍 搜索查询：{query}")
    logger.info("\n📌 匹配结果:")
    logger.info("─" * 60)

    if results["documents"] and results["documents"][0]:
        for i, (id, doc, meta, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                strict=True,
            ),
            1,
        ):
            file_name = meta.get("file_name", "unknown")
            title = meta.get("title", "未知")
            # Chroma 返回的是余弦距离，转换为相似度
            similarity = 1 - distance if distance is not None else None

            if show_score and similarity is not None:
                logger.info(f"[{i}]  id:{id}  📄 {title}")
                logger.info(f"    📁 来源：{file_name}")
                logger.info(f"    📊 相似度：{similarity:.4f} (距离：{distance:.4f})")
            else:
                logger.info(f"[{i}]  id:{id}   {file_name}")
            logger.info(f"    📝 {doc[:150]}...")
            logger.info("")
    else:
        logger.info("⚠️  暂无结果，请先运行索引命令添加知识。")

    logger.info("─" * 60)
    logger.info("✅ 搜索完成")
