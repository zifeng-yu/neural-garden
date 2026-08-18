import json
from dataclasses import asdict, dataclass

from src.config.config import (
    CHROMA_CONCEPT_TABLE_NAME,
    CHROMA_KNOWLEDGE_TABLE_NAME,
)
from src.get_chroma_collection import get_collection


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
    collection.upsert(
        ids=[knowledgeUnitDTO.id],
        embeddings=[knowledgeUnitDTO.embedding],
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
    collection.upsert(
        ids=[conceptDTO.id],
        embeddings=[conceptDTO.embedding],
        documents=[conceptDTO.normalized_concept],
    )
