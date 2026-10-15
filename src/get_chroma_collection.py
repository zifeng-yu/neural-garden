from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from src.config.config import (
    CHROMA_DISTANCE_METRIC,
    PERSIST_DIRECTORY,
)


def get_collection(collection_name: str) -> Collection:
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    return client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": CHROMA_DISTANCE_METRIC}
    )


def get_existing_collection(collection_name: str) -> Collection:
    """读取已有collection；源目录或collection缺失时失败，不新建空集合。"""
    if not Path(PERSIST_DIRECTORY).is_dir():
        raise FileNotFoundError(f"Chroma源目录不存在：{PERSIST_DIRECTORY}")
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    return client.get_collection(name=collection_name)
