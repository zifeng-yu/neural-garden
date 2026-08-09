import os
from dataclasses import asdict, dataclass

import chromadb
from chromadb.api.models.Collection import Collection

from src.config.config import (
    CHROMA_CONCEPT_TABLE_NAME,
    CHROMA_KNOWLEDGE_TABLE_NAME,
    PERSIST_DIRECTORY,
)
from src.get_chroma_collection import get_collection


def save_knowlege():
    pass


@dataclass
class ConceptMetadata:
    """metadata"""

    source: list[str]

    def to_dict(self) -> dict:
        """转换为字典（用于存储）"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(source=list(data.get("source", [])))


@dataclass
class ConceptDO:
    """concept结构体"""

    id: str
    embedding: list[float]
    conceptName: str
    metadata: ConceptMetadata


def save_concept(conceptDO: ConceptDO):
    collection = get_collection(CHROMA_CONCEPT_TABLE_NAME)
    collection.upsert(
        ids=[conceptDO.id],
        embeddings=[conceptDO.embedding],
        documents=[conceptDO.conceptName],
        metadatas=[conceptDO.metadata.to_dict()],
    )
