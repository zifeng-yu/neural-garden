"""
Neural Garden - 知识单元索引器
Lesson 02.1: 生产级版本（支持增量更新）

功能：
- 批量导入 pilot/ 目录下所有 .md 文件
- 知识单元提取（标题、摘要、关键词）
- 调用 DashScope Embedding API 向量化
- 增量更新检测（基于 content_hash，避免重复索引）
- 支持 --reset 参数清空向量库
"""

import json
import logging
import os

import chromadb

import src.config.logging_config as logging_config
from src.config.config import CHROMA_TABLE_NAME, PERSIST_DIRECTORY, PILOT_DATASET_PATH
from src.embedding.getEmbedding import get_embedding
from src.knowledge.knowledgeUnit import extract_knowledge_unit
from src.util.getHashValue import get_hash_value as hash
from src.vector_store.reset import resetDB

logger = logging.getLogger(__name__)


def load_document(file_path: str) -> str:
    """
    读取 Markdown 文件内容

    Args:
        file_path: 文件路径

    Returns:
        文件内容字符串
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def index_file(file_path: str, persist_dir: str):
    """
    将单个文件导入 Chroma 向量库

    Args:
        file_path: Markdown 文件路径
        persist_dir: Chroma 持久化目录
        api_key: DashScope API Key
    """
    # 1. 读取文件
    content = load_document(file_path)
    source = os.path.basename(file_path)

    logger.info(f"📄 正在索引：{source}")

    # 2. 初始化 Chroma 客户端
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=CHROMA_TABLE_NAME, metadata={"hnsw:space": "cosine"}
    )

    # 3. 检查是否已存在 uuid = 文件名hash , content_hash = 文件内容hash
    #   文件名变换 一定属于新增，文章名没变 看内容 是否有变化
    uuid = hash(source)
    content_hash = hash(content)
    existing = collection.get(ids=[uuid], include=["metadatas"])
    if existing["ids"] and existing["metadatas"][0].get("content_hash") == content_hash:
        logger.info(f"⏭️  已存在，跳过：{source}")
        return

    # 4. 获取知识单元
    knowledgeUnit = extract_knowledge_unit(content=content, source=source, uuid=uuid)
    if knowledgeUnit is None:
        logger.info(f"知识提取失败，source {source}")
        return

    # 5. 调用 Embedding API
    logger.info("🔢 正在生成向量...")
    embedding = get_embedding(knowledgeUnit.to_embedding_text())
    if embedding is None:
        logger.info(f"⚠️  跳过 {source}（Embedding 失败）")
        return

    # logger.info(f"向量返回：{embedding}")
    # 6. 添加到向量库
    collection.upsert(
        ids=[knowledgeUnit.unit_id],
        embeddings=[embedding],
        documents=[knowledgeUnit.to_embedding_text()],  # 存储向量化文本
        metadatas=[
            {
                "source": source,
                "title": knowledgeUnit.title,
                "keywords": json.dumps(knowledgeUnit.keywords),
                "full_content": content,  # 可选：存储全文用于后续检索
                "content_hash": content_hash,
            }
        ],
    )

    logger.info(f"✅ 索引完成：{source} , collection size {collection.count()}")


def index_directory(dir_path: str, persist_dir: str):
    """
    批量索引目录下所有 .md 文件

    Args:
        dir_path: 目录路径
        persist_dir: Chroma 持久化目录
    """
    md_files = [f for f in os.listdir(dir_path) if f.endswith(".md")]

    if not md_files:
        print(f"⚠️  未找到 .md 文件：{dir_path}")
        return

    logger.info(f"📂 发现 {len(md_files)} 个 Markdown 文件\n")

    for filename in md_files:
        file_path = os.path.join(dir_path, filename)
        index_file(file_path, persist_dir)

    logger.info(f"\n✅ 批量索引完成，共 {len(md_files)} 个文件")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if "--resetDB" in args:
        resetDB()

    # 配置路径
    base_dir = os.path.dirname(os.path.dirname(__file__))
    persist_dir = os.path.join(base_dir, PERSIST_DIRECTORY)
    pilot_dir = os.path.join(base_dir, PILOT_DATASET_PATH)

    # 默认索引 pilot/ 目录
    logger.info("🌱 Neural Garden 索引器（骨架版本）")
    logger.info(f"📂 索引目录：{pilot_dir}")
    logger.info(f"💾 存储目录：{persist_dir}\n")

    index_directory(pilot_dir, persist_dir)

    logger.info("\n" + "=" * 40)
    logger.info("💡 提示：运行以下命令搜索知识：")
    logger.info(f"   python -m src.search <查询内容>")
