from dataclasses import dataclass
from enum import Enum

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO


class RelationSource(str, Enum):
    # 文档内
    DOCUMENT_INTERNAL = "document_internal"
    # 跨文档
    CROSS_DOCUMENT = "cross_document"


@dataclass
class ConceptRelations(BaseDO):
    source_concept: str
    target_concept: str
    relation: str
    relation_source: str
    confidence: float


def insert_concept_relation(
    source_concept: str,
    target_concept: str,
    relation: str,
    relation_source: str,
    confidence: float | None = None,
):
    with get_sqlite_connection() as conn:

        cursor = conn.execute(
            """
            INSERT INTO concept_relations
            (
                source_concept,
                target_concept,
                relation,
                relation_source,
                confidence
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                source_concept,
                target_concept,
                relation,
                relation_source,
                confidence,
            ),
        )

        conn.commit()

        return cursor.lastrowid
