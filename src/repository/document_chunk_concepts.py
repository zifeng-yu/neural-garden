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
    conn: sqlite3.Connection,
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
