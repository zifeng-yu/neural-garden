from src.config.config import (
    CHROMA_CONCEPT_TABLE_NAME,
    CHROMA_KNOWLEDGE_TABLE_NAME,
)
from src.get_chroma_collection import get_collection


def delete_by_id_knowledge(ids: list[str]):
    if not ids:
        return
    collection = get_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
    collection.delete(ids=ids)


def delete_by_normalized_concet_hash_concept(ids: list[str]):
    if not ids:
        return
    collection = get_collection(CHROMA_CONCEPT_TABLE_NAME)
    collection.delete(ids=ids)
