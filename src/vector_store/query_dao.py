import logging

from src.config.config import (
    CHROMA_CONCEPT_TABLE_NAME,
    CHROMA_KNOWLEDGE_TABLE_NAME,
)
from src.embedding.getEmbedding import get_embedding
from src.get_chroma_collection import get_collection
from src.vector_store.save_dao import ConceptDTO

logger = logging.getLogger(__name__)


def get_by_id_concept(id: str) -> ConceptDTO | None:
    collection = get_collection(CHROMA_CONCEPT_TABLE_NAME)
    result = collection.get(ids=[id], include=["documents", "metadatas", "embeddings"])
    logger.info(f"向量搜索 概念id hash  {id} : {result}")
    if not result["ids"]:
        return None
    embeddings = result["embeddings"]
    if embeddings is None or len(embeddings) == 0:
        return None
    docs = result["documents"]
    if docs is None or len(docs) == 0:
        return None
    return ConceptDTO(
        id=result["ids"][0],
        embedding=list(embeddings[0]),
        normalized_concept=docs[0],
    )


def search_by_threshold_concept(
    query: str, threshold: float = 0.2, top_k: int = 1
) -> list[ConceptDTO]:
    collection = get_collection(CHROMA_CONCEPT_TABLE_NAME)
    embedding = get_embedding(query)
    if embedding is None:
        logger.error("❌ Embedding 生成失败")
        return []

    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    concept_results = []

    documents = results.get("documents")
    distances = results.get("distances")

    if not documents or not distances:
        return []
    for doc, distance in zip(documents[0], distances[0], strict=True):
        similarity = 1 - distance
        if similarity > threshold:
            concept_results.append(
                ConceptDTO(
                    id="",
                    embedding=[],
                    normalized_concept=doc,
                )
            )

    return concept_results


def log_concept_collection_size():
    collection = get_collection(CHROMA_CONCEPT_TABLE_NAME)
    logger.info(f"log_concept_collection_size {collection.count()}")


def log_knowledgeUnit_collection_size():
    collection = get_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
    logger.info(f"log_knowledgeUnit_collection_size {collection.count()}")
