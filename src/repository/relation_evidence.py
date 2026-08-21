import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO

TABLE_NAME = "relation_evidence"


class EvidenceRoleEnum(str, Enum):
    BOTH = "both"
    SOURCE = "source"
    TARGET = "target"


@dataclass
class RelationEvidence(BaseDO):
    relation_id: int
    document_id: int
    document_chunk_id: int
    evidence_role: EvidenceRoleEnum = EvidenceRoleEnum.BOTH


def insert_relation_evidence(
    relation_id: int,
    document_id: int,
    document_chunk_id: int,
    evidence_role: EvidenceRoleEnum,
):
    with get_sqlite_connection() as conn:

        cursor = conn.execute(
            """
            INSERT INTO relation_evidence
            (
                relation_id,
                document_id,
                document_chunk_id,
                evidence_role
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT (
                relation_id,
                document_id,
                document_chunk_id,
                evidence_role
            )
            DO UPDATE SET
                id = id
            RETURNING id
            """,
            (
                relation_id,
                document_id,
                document_chunk_id,
                evidence_role.value,
            ),
        )
        id = cursor.fetchone()[0]

        return id


def query_by_document_id(document_id: int) -> list[RelationEvidence]:
    with get_sqlite_connection() as conn:
        rows = conn.execute(
            f"""
                select * from {TABLE_NAME} where document_id = ?
            """,
            (document_id,),
        ).fetchall()
        if not rows:
            return []

        result = []
        for row in rows:
            data = dict(row)
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            data["evidence_role"] = EvidenceRoleEnum(data["evidence_role"])
            result.append(RelationEvidence(**data))

        return result


def _insert_relation_evidence(
    conn: sqlite3.Connection,
    relation_id: int,
    document_id: int,
    document_chunk_id: int,
    evidence_role: EvidenceRoleEnum,
):

    cursor = conn.execute(
        """
        INSERT INTO relation_evidence
        (
            relation_id,
            document_id,
            document_chunk_id,
            evidence_role
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            relation_id,
            document_id,
            document_chunk_id,
            evidence_role.value,
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("insert document failed")
    return cursor.lastrowid


def _query_by_document_id(
    conn: sqlite3.Connection, document_id: int
) -> list[RelationEvidence]:
    rows = conn.execute(
        f"""
            select * from {TABLE_NAME} where document_id = ?
        """,
        (document_id,),
    ).fetchall()
    if not rows:
        return []

    result = []
    for row in rows:
        data = dict(row)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["evidence_role"] = EvidenceRoleEnum(data["evidence_role"])
        result.append(RelationEvidence(**data))

    return result


def copy_relation_evidence(
    conn: sqlite3.Connection,
    need_document_id: int,
    new_document_id: int,
    old_new_chunk_id_map: dict[int, int],
) -> list[int]:
    relation_evidence_do_list = _query_by_document_id(conn, need_document_id)
    if not relation_evidence_do_list:
        return []
    new_relation_evidence_ids = []
    for relation_evidence_do in relation_evidence_do_list:
        id = _insert_relation_evidence(
            conn,
            relation_evidence_do.relation_id,
            new_document_id,
            old_new_chunk_id_map[relation_evidence_do.document_chunk_id],
            relation_evidence_do.evidence_role,
        )
        new_relation_evidence_ids.append(id)
    return new_relation_evidence_ids


def delete_by_document_id(conn: sqlite3.Connection, document_id: int):
    conn.execute(
        f"""
                delete from {TABLE_NAME} where document_id = ?
            """,
        (document_id,),
    )


def query_by_relation_id(
    conn: sqlite3.Connection, relation_id: int
) -> list[RelationEvidence]:
    if conn is not None:
        return _query_by_relation_id(conn, relation_id)
    with get_sqlite_connection() as new_conn:
        return _query_by_relation_id(new_conn, relation_id)


def _query_by_relation_id(
    conn: sqlite3.Connection, relation_id: int
) -> list[RelationEvidence]:
    rows = conn.execute(
        f"""
            select * from {TABLE_NAME} where relation_id = ?
        """,
        (relation_id,),
    ).fetchall()
    if not rows:
        return []

    result = []
    for row in rows:
        data = dict(row)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["evidence_role"] = EvidenceRoleEnum(data["evidence_role"])
        result.append(RelationEvidence(**data))

    return result
