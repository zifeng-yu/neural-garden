"""
Neural Garden - 知识单元索引器
Lesson 01: 骨架版本（支持单文件导入）

完整功能将在 Lesson 02 实现：
- 批量导入 pilot/ 目录下所有 .md 文件
- 知识单元提取（标题、摘要、关键词）
- 调用 DashScope Embedding API
"""

import hashlib
import os

import chromadb

from config import CHROMA_TABLE_NAME, PERSIST_DIRECTORY, PILOT_DATASET_PATH
from src.embedding.getEmbedding import get_embedding
from src.vector_store.reset import reset


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


def generate_id(content: str) -> str:
    """
    生成内容哈希 ID（避免重复）

    Args:
        content: 文件内容

    Returns:
        16 位哈希字符串
    """
    return hashlib.md5(content.encode()).hexdigest()[:16]


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
    doc_id = generate_id(content)
    source = os.path.basename(file_path)

    print(f"📄 正在索引：{source}")

    # 2. 初始化 Chroma 客户端
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(CHROMA_TABLE_NAME)

    # 3. 检查是否已存在
    # existing = collection.get(ids=[doc_id])
    # if existing["ids"]:
    #     print(f"⏭️  已存在，跳过：{source}")
    #     return

    # 4. 调用 Embedding API
    print(f"🔢 正在生成向量...")
    embedding = get_embedding(content)

    if embedding is None:
        print(f"⚠️  跳过 {source}（Embedding 失败）")
        return
    # print(f"向量返回：{embedding}")
    # 5. 添加到向量库
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{"source": source}],
    )

    print(f"✅ 索引完成：{source} , collection size {collection.count()}")


def index_directory(dir_path: str, persist_dir: str):
    """
    批量索引目录下所有 .md 文件

    Args:
        dir_path: 目录路径
        persist_dir: Chroma 持久化目录
        api_key: DashScope API Key
    """
    md_files = [f for f in os.listdir(dir_path) if f.endswith(".md")]

    if not md_files:
        print(f"⚠️  未找到 .md 文件：{dir_path}")
        return

    print(f"📂 发现 {len(md_files)} 个 Markdown 文件\n")

    # 测试阶段，先reset chromaDB
    reset()
    for filename in md_files:
        file_path = os.path.join(dir_path, filename)
        index_file(file_path, persist_dir)

    print(f"\n✅ 批量索引完成，共 {len(md_files)} 个文件")


if __name__ == "__main__":

    # 配置路径
    base_dir = os.path.dirname(os.path.dirname(__file__))
    persist_dir = os.path.join(base_dir, PERSIST_DIRECTORY)
    pilot_dir = os.path.join(base_dir, PILOT_DATASET_PATH)

    # 默认索引 pilot/ 目录
    print("🌱 Neural Garden 索引器（骨架版本）")
    print(f"📂 索引目录：{pilot_dir}")
    print(f"💾 存储目录：{persist_dir}\n")

    index_directory(pilot_dir, persist_dir)

    print("\n" + "=" * 40)
    print("💡 提示：运行以下命令搜索知识：")
    print(f"   python -m src.search <查询内容>")
