from dataclasses import dataclass
from enum import Enum

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO


class EvidenceRoleEnum(str, Enum):
    BOTH = "both"
    SOURCE = "source"
    TARGET = "target"


@dataclass
class RelationEvidence(BaseDO):
    relation_id: int
    doc_id: int
    chunk_id: int
    evidence_role: str = EvidenceRoleEnum.BOTH.value


def insert_relation_evidence(
    relation_id: int,
    doc_id: int,
    chunk_id: int,
    evidence_role: EvidenceRoleEnum,
):
    with get_sqlite_connection() as conn:

        cursor = conn.execute(
            """
            INSERT INTO relation_evidence
            (
                relation_id,
                doc_id,
                chunk_id,
                evidence_role
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                relation_id,
                doc_id,
                chunk_id,
                evidence_role.value,
            ),
        )

        conn.commit()

        return cursor.lastrowid
