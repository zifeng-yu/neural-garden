from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO

TABLE_NAME = "concept_relations"


class RelationType(str, Enum):
    # 文档内
    DOCUMENT_INTERNAL = "document_internal"
    # 跨文档
    CROSS_DOCUMENT = "cross_document"


@dataclass
class ConceptRelations(BaseDO):
    source_concept: str
    target_concept: str
    relation: str
    relation_type: RelationType
    confidence: float


def insert_concept_relation(
    source_concept: str,
    target_concept: str,
    relation: str,
    relation_type: RelationType,
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
                relation_type,
                confidence
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (
                source_concept,
                target_concept,
                relation,
                relation_type
            )
            DO UPDATE SET
                id = id
            RETURNING id
            """,
            (
                source_concept,
                target_concept,
                relation,
                relation_type.value,
                confidence,
            ),
        )
        id = cursor.fetchone()[0]

        return id


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
            data["relation_type"] = RelationType(data["relation_type"])
            result.append(ConceptRelations(**data))

        return result


def delete_by_id(id: int):
    with get_sqlite_connection() as conn:
        conn.execute(
            f"""
                    delete from {TABLE_NAME} where id = ?
                """,
            (id,),
        )
