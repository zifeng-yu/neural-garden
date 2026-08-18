from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO

TABLE_NAME = "concept_relations"


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
    relation_source: RelationSource
    confidence: float


def insert_concept_relation(
    source_concept: str,
    target_concept: str,
    relation: str,
    relation_source: RelationSource,
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
                relation_source.value,
                confidence,
            ),
        )

        conn.commit()

        return cursor.lastrowid


def query_all() -> list[ConceptRelations]:
    with get_sqlite_connection() as conn:

        rows = conn.execute(
            f"""
                    select * from {TABLE_NAME}
                    """,
        ).fetchall()
        if not rows:
            return []

        result = []
        for row in rows:
            data = dict(row)
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            result.append(ConceptRelations(**data))

        return result
