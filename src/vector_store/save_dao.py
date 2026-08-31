import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from src.config.config import (
    CHROMA_CONCEPT_TABLE_NAME,
    CHROMA_INSIGHT_TABLE_NAME,
    CHROMA_KNOWLEDGE_TABLE_NAME,
)
from src.get_chroma_collection import get_collection

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeUnitMetadata:
    file_name: str
    document_id: int
    document_chunk_id: int
    title: str
    keywords: list[str]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["keywords"] = json.dumps(self.keywords, ensure_ascii=False)
        return data


@dataclass
class KnowledgeUnitDTO:
    """concept结构体"""

    # chunk_know_id
    id: str
    embedding: list[float]
    text: str
    metadata: KnowledgeUnitMetadata


def save_knowlege(knowledgeUnitDTO: KnowledgeUnitDTO):
    collection = get_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
    embeddings: list[Sequence[float]] = [knowledgeUnitDTO.embedding]
    collection.upsert(
        ids=[knowledgeUnitDTO.id],
        embeddings=embeddings,
        documents=[knowledgeUnitDTO.text],
        metadatas=[knowledgeUnitDTO.metadata.to_dict()],
    )


@dataclass
class ConceptDTO:
    """concept结构体"""

    # hash(normalized_concept)
    id: str
    embedding: list[float]
    normalized_concept: str


def save_concept(conceptDTO: ConceptDTO):
    collection = get_collection(CHROMA_CONCEPT_TABLE_NAME)
    embeddings: list[Sequence[float]] = [conceptDTO.embedding]
    collection.upsert(
        ids=[conceptDTO.id],
        embeddings=embeddings,
        documents=[conceptDTO.normalized_concept],
    )


@dataclass
class InsightMetadata:
    insight_id: int
    source_type: str
    source_status: str
    document_id: int | None
    document_chunk_id: int | None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class InsightDTO:
    id: str
    embedding: list[float]
    embedding_text: str
    metadata: InsightMetadata


def save_insight(insightDTO: InsightDTO):
    collection = get_collection(CHROMA_INSIGHT_TABLE_NAME)
    embeddings: list[Sequence[float]] = [insightDTO.embedding]
    collection.upsert(
        ids=[insightDTO.id],
        embeddings=embeddings,
        documents=[insightDTO.embedding_text],
        metadatas=[insightDTO.metadata.to_dict()],
    )


def update_insight_metadata(insight_id: str, insight_metadata: InsightMetadata):
    collection = get_collection(CHROMA_INSIGHT_TABLE_NAME)
    collection.update(
        ids=[insight_id],
        metadatas=[insight_metadata.to_dict()],
    )


def log_insight_collection_count():
    collection = get_collection(CHROMA_INSIGHT_TABLE_NAME)
    logger.info(f"insight 集合大小 {collection.count()}")
