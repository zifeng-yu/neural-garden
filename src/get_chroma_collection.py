import os

import chromadb
from chromadb.api.models.Collection import Collection

from src.config.config import (
    PERSIST_DIRECTORY,
)


def get_collection(collection_name: str) -> Collection:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    persist_dir = os.path.join(base_dir, PERSIST_DIRECTORY)
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
