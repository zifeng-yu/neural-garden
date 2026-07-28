"""
Neural Garden - 搜索入口
Lesson 01: Hello World 骨架版本
"""

import logging
import os

import chromadb

import src.config.logging_config as logging_config
from src.config.config import CHROMA_TABLE_NAME, PERSIST_DIRECTORY
from src.embedding.getEmbedding import get_embedding

logger = logging.getLogger(__name__)


def search(query: str, top_k: int = 3):
    """
    向量搜索入口

    Args:
        query: 搜索查询
        top_k: 返回结果数量
    """

    # 1. 初始化 Chroma 客户端
    base_dir = os.path.dirname(os.path.dirname(__file__))
    persist_dir = os.path.join(base_dir, PERSIST_DIRECTORY)
    client = chromadb.PersistentClient(path=persist_dir)

    # 2. 获取集合
    collection = client.get_collection(CHROMA_TABLE_NAME)
    logger.info(f"表集合大小 {collection.count()}")

    # 3. 查询（骨架版本，后续课程实现完整 Embedding 流程）
    embedding = get_embedding(query)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    logger.info(f"查询结果 {results}")
    # 4. 格式化输出
    log_results(results, query)


def log_results(results, query):
    """格式化输出搜索结果"""
    logger.info(f"\n🔍 搜索查询：{query}")
    logger.info("\n📌 匹配结果 (Top 3):")
    logger.info("─" * 40)

    if results["documents"] and results["documents"][0]:
        for i, (doc, meta, distances) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            1,
        ):
            source = meta.get("source", "unknown")
            logger.info(f"[{i}] {source}  距离： {distances}")
            logger.info(f"    {doc[:150]}...")
    else:
        logger.info("暂无结果，请先运行索引命令添加知识。")

    logger.info("─" * 40)
    logger.info("✅ 搜索完成")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        search(query)
    else:
        logger.info("用法：python -m src.search <查询内容>")
