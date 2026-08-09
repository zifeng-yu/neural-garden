import logging

import src.config.logging_config as logging_config
from src.config.config import (
    CHROMA_CONCEPT_TABLE_NAME,
    CHROMA_KNOWLEDGE_TABLE_NAME,
)
from src.embedding.getEmbedding import get_embedding
from src.get_chroma_collection import get_collection
from src.vector_store.save_dao import ConceptDO, ConceptMetadata

logger = logging.getLogger(__name__)


def get_by_id_concept(id: str) -> ConceptDO | None:
    collection = get_collection(CHROMA_CONCEPT_TABLE_NAME)
    result = collection.get(ids=[id], include=["documents", "metadatas", "embeddings"])
    logger.info(f"向量搜索 概念id hash  {id} : {result}")
    if not result["ids"]:
        return None
    return ConceptDO(
        id=result["ids"][0],
        embedding=result["embeddings"][0],
        conceptName=result["documents"][0],
        metadata=ConceptMetadata.from_dict(result["metadatas"][0]),
    )


def search_by_threshold_concept(
    query: str, threshold: float = 0.85, top_k: int = 1
) -> list[ConceptDO] | None:
    collection = get_collection(CHROMA_CONCEPT_TABLE_NAME)
    embedding = get_embedding(query)
    if embedding is None:
        logger.error("❌ Embedding 生成失败")
        return None

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    concept_results = []

    documents = results.get("documents")
    metadatas = results.get("metadatas")
    distances = results.get("distances")

    if not documents or not metadatas or not distances:
        return []
    for doc, meta, distance in zip(documents[0], metadatas[0], distances[0]):
        similarity = 1 - distance
        logger.info(f" 概念 {query} : 最高相似度 概念 {doc} 相似度 {similarity}")
        if similarity > threshold:
            concept_results.append(
                ConceptDO(
                    id="",
                    embedding=[],
                    conceptName=doc,
                    metadata=ConceptMetadata.from_dict(meta),
                )
            )

    return concept_results
