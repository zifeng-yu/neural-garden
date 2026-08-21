import sqlite3
from dataclasses import dataclass
from datetime import datetime

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO

TABLE_NAME = "document_chunk_concepts"


@dataclass
class DocumentChunkConceptsDO(BaseDO):
    document_id: int
    document_chunk_id: int
    concept: str
    normalized_concept: str
    normalized_concept_hash: str


def insert_concept(
    conn: sqlite3.Connection | None,
    document_id: int,
    document_chunk_id: int,
    concept: str,
    normalized_concept: str,
    normalized_concept_hash: str,
) -> int:
    if conn is not None:
        return _insert_concept(
            conn,
            document_id,
            document_chunk_id,
            concept,
            normalized_concept,
            normalized_concept_hash,
        )
    with get_sqlite_connection() as new_conn:
        return _insert_concept(
            new_conn,
            document_id,
            document_chunk_id,
            concept,
            normalized_concept,
            normalized_concept_hash,
        )


def _insert_concept(
    conn: sqlite3.Connection,
    document_id: int,
    document_chunk_id: int,
    concept: str,
    normalized_concept: str,
    normalized_concept_hash: str,
) -> int:
    cursor = conn.execute(
        f"""
        INSERT INTO {TABLE_NAME} (
            document_id,
            document_chunk_id,
            concept,
            normalized_concept,
            normalized_concept_hash
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            document_id,
            document_chunk_id,
            concept,
            normalized_concept,
            normalized_concept_hash,
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("insert document_chunk_concepts failed")

    return cursor.lastrowid


def query_by_normalized_concept(
    normalized_concept: str,
) -> list[DocumentChunkConceptsDO]:
    with get_sqlite_connection() as conn:

        rows = conn.execute(
            f"""
                    select * from {TABLE_NAME} where normalized_concept = ?
                    """,
            (normalized_concept,),
        ).fetchall()
        if not rows:
            return []

        result = []
        for row in rows:
            data = dict(row)
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            result.append(DocumentChunkConceptsDO(**data))

        return result


def query_by_document_id(
    document_id: int,
) -> list[DocumentChunkConceptsDO]:
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
            result.append(DocumentChunkConceptsDO(**data))

        return result


def query_by_not_document_id_and_in_normalized_concepts(
    document_id: int, normalized_concepts: list[str]
) -> list[DocumentChunkConceptsDO]:
    if not normalized_concepts:
        return []
    placeholders = ",".join("?" for _ in normalized_concepts)
    with get_sqlite_connection() as conn:

        rows = conn.execute(
            f"""
                    select * from {TABLE_NAME} where document_id != ? and normalized_concept in({placeholders})
                    """,
            (
                document_id,
                *normalized_concepts,
            ),
        ).fetchall()
        if not rows:
            return []

        result = []
        for row in rows:
            data = dict(row)
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            result.append(DocumentChunkConceptsDO(**data))

        return result


def _query_by_document_id(
    conn: sqlite3.Connection,
    document_id: int,
) -> list[DocumentChunkConceptsDO]:
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
        result.append(DocumentChunkConceptsDO(**data))

    return result


def query_by_chunk_id(
    chunk_id: int,
) -> list[DocumentChunkConceptsDO]:
    with get_sqlite_connection() as conn:

        rows = conn.execute(
            f"""
                    select * from {TABLE_NAME} where document_chunk_id = ?
                    """,
            (chunk_id,),
        ).fetchall()
        if not rows:
            return []

        result = []
        for row in rows:
            data = dict(row)
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            result.append(DocumentChunkConceptsDO(**data))

        return result


def query_all() -> list[DocumentChunkConceptsDO]:
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
            result.append(DocumentChunkConceptsDO(**data))

        return result


def copy_documents_chunk_concepts(
    conn: sqlite3.Connection,
    need_document_id: int,
    new_document_id: int,
    old_new_chunk_id_map: dict[int, int],
) -> list[int]:
    chunk_concepts_list = _query_by_document_id(conn, need_document_id)
    if not chunk_concepts_list:
        return []
    new_chunk_concepts_ids = []
    for chunk_concepts in chunk_concepts_list:
        new_id = insert_concept(
            conn,
            new_document_id,
            old_new_chunk_id_map[chunk_concepts.document_chunk_id],
            chunk_concepts.concept,
            chunk_concepts.normalized_concept,
            chunk_concepts.normalized_concept_hash,
        )
        new_chunk_concepts_ids.append(new_id)
    return new_chunk_concepts_ids


def delete_by_document_id(conn: sqlite3.Connection, document_id: int):
    conn.execute(
        f"""
                delete from {TABLE_NAME} where document_id = ?
            """,
        (document_id,),
    )
