import logging
from dataclasses import dataclass, field

import src.config.logging_config as logging_config
from src.get_sqlite_connection import get_sqlite_connection
from src.repository.document_chunk_concepts import insert_concept
from src.repository.document_chunk_knowledge_units import insert_knowledge_unit
from src.repository.document_chunks import insert_document_chunk
from src.repository.documents import insert_document

logger = logging.getLogger(__name__)


@dataclass
class InsertDoc:
    file_name: str
    file_name_hash: str
    content_hash: str


@dataclass
class InsertKnowledgeUnit:
    title: str
    summary: str
    keywords: list[str]
    embedding_text: str


@dataclass
class InsertConcepts:
    concept: str
    normalized_concept: str
    normalized_concept_hash: str


@dataclass
class InsertChunk:
    split_no: int
    content: str
    content_hash: str
    know_unit: InsertKnowledgeUnit | None = None
    concepts: list[InsertConcepts] = field(default_factory=list)


@dataclass
class InsertDocumentDomain:
    insert_doc: InsertDoc
    chunks: list[InsertChunk]


def save_documents_domain(document_domain: InsertDocumentDomain):
    try:
        with get_sqlite_connection() as conn:
            doc_id = insert_document(
                conn,
                document_domain.insert_doc.file_name,
                document_domain.insert_doc.file_name_hash,
                document_domain.insert_doc.content_hash,
            )
            chunk_ids = []
            knowledge_unit_ids = []
            for chunk in document_domain.chunks:
                chunk_id = insert_document_chunk(
                    conn, doc_id, chunk.split_no, chunk.content, chunk.content_hash
                )
                chunk_ids.append(chunk_id)

                if chunk.know_unit is not None:
                    knowledge_unit_id = insert_knowledge_unit(
                        conn,
                        doc_id,
                        chunk_id,
                        chunk.know_unit.title,
                        chunk.know_unit.summary,
                        chunk.know_unit.keywords,
                        chunk.know_unit.embedding_text,
                    )
                    knowledge_unit_ids.append(knowledge_unit_id)

                for concept_domain in chunk.concepts:
                    insert_concept(
                        conn,
                        doc_id,
                        chunk_id,
                        concept_domain.concept,
                        concept_domain.normalized_concept,
                        concept_domain.normalized_concept_hash,
                    )
            return {
                "document_id": doc_id,
                "knowledge_unit_ids": knowledge_unit_ids,
                "chunk_ids": chunk_ids,
            }
    except Exception:
        logger.exception("save_documents_domain 出现错误，事务回滚 ")
        raise
