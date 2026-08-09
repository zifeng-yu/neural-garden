"""
Neural Garden - 搜索入口
Lesson 03: 向量搜索入门（增强版）
"""

import logging
import os

import chromadb

import src.config.logging_config as logging_config
from src.config.config import CHROMA_KNOWLEDGE_TABLE_NAME, PERSIST_DIRECTORY
from src.embedding.getEmbedding import get_embedding

logger = logging.getLogger(__name__)


def search(query: str, top_k: int = 3, show_score: bool = True):
    """
    向量搜索入口（增强版）

    Args:
        query: 搜索查询
        top_k: 返回结果数量
        show_score: 是否显示相似度分数
    """
    # 1. 初始化 Chroma 客户端
    base_dir = os.path.dirname(os.path.dirname(__file__))
    persist_dir = os.path.join(base_dir, PERSIST_DIRECTORY)
    client = chromadb.PersistentClient(path=persist_dir)

    # 2. 获取集合
    collection = client.get_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
    logger.info(f"表集合大小 {collection.count()}")

    # 3. 生成查询向量并搜索
    embedding = get_embedding(query)
    if embedding is None:
        logger.error("❌ Embedding 生成失败")
        return

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # 4. 格式化输出（带相似度）
    log_results(results, query, show_score=show_score)


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
        for i, (doc, meta, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            1,
        ):
            source = meta.get("source", "unknown")
            title = meta.get("title", "未知")
            # Chroma 返回的是余弦距离，转换为相似度
            similarity = 1 - distance if distance is not None else None

            if show_score and similarity is not None:
                logger.info(f"[{i}] 📄 {title}")
                logger.info(f"    📁 来源：{source}")
                logger.info(f"    📊 相似度：{similarity:.4f} (距离：{distance:.4f})")
            else:
                logger.info(f"[{i}] {source}")
            logger.info(f"    📝 {doc[:150]}...")
            logger.info("")
    else:
        logger.info("⚠️  暂无结果，请先运行索引命令添加知识。")

    logger.info("─" * 60)
    logger.info("✅ 搜索完成")


def search_with_filter(
    query: str, filter_field: str, filter_value: str, top_k: int = 3
):
    """
    带过滤条件的搜索（Lesson 03 新增）

    Args:
        query: 搜索查询
        filter_field: 过滤字段名（如 "source"）
        filter_value: 过滤值
        top_k: 返回结果数量
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    persist_dir = os.path.join(base_dir, PERSIST_DIRECTORY)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(CHROMA_KNOWLEDGE_TABLE_NAME)

    embedding = get_embedding(query)
    if embedding is None:
        logger.error("❌ Embedding 生成失败")
        return

    # Chroma 的 where 过滤语法
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where={filter_field: filter_value},
        include=["documents", "metadatas", "distances"],
    )

    logger.info(f"\n🔍 搜索：「{query}」（限定 {filter_field}={filter_value}）\n")
    log_results(results, query)


def search_by_threshold(query: str, threshold: float = 0.7, top_k: int = 20):
    """
    按阈值搜索（Lesson 03 课后练习实现）

    Args:
        query: 搜索关键词
        threshold: 相似度阈值（默认 0.7）
        top_k: 初始搜索数量（先获取较多结果，再过滤）

    Returns:
        只返回相似度 > threshold 的结果
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    persist_dir = os.path.join(base_dir, PERSIST_DIRECTORY)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(CHROMA_KNOWLEDGE_TABLE_NAME)

    embedding = get_embedding(query)
    if embedding is None:
        logger.error("❌ Embedding 生成失败")
        return

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # 过滤：只保留相似度 > threshold 的结果
    filter_threshold_documents = []
    filter_threshold_metadatas = []
    filter_threshold_distances = []

    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        similarity = 1 - distance
        if similarity > threshold:
            filter_threshold_documents.append(doc)
            filter_threshold_metadatas.append(meta)
            filter_threshold_distances.append(distance)

    filter_threshold_result = {
        "documents": [filter_threshold_documents],
        "metadatas": [filter_threshold_metadatas],
        "distances": [filter_threshold_distances],
    }

    logger.info(f"\n🔍 阈值搜索：「{query}」（相似度 > {threshold}）\n")
    log_results(filter_threshold_result, query)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        search(query, show_score=True)
    else:
        logger.info("用法：python -m src.search <查询内容>")
        logger.info("示例：python -m src.search 负利率")
