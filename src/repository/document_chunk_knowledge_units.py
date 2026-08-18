import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO

TABLE_NAME = "document_chunk_knowledge_units"


@dataclass
class DocumentChunkKnowledgeUnits(BaseDO):
    document_id: int
    document_chunk_id: int
    title: str
    summary: str
    keywords: list[str]
    embedding_text: str


def insert_knowledge_unit(
    conn: sqlite3.Connection,
    document_id: int,
    document_chunk_id: int,
    title: str,
    summary: str,
    keywords: list[str],
    embedding_text: str,
) -> int:
    if conn is not None:
        return _insert_knowledge_unit(
            conn,
            document_id,
            document_chunk_id,
            title,
            summary,
            keywords,
            embedding_text,
        )
    with get_sqlite_connection() as new_conn:
        return _insert_knowledge_unit(
            new_conn,
            document_id,
            document_chunk_id,
            title,
            summary,
            keywords,
            embedding_text,
        )


def _insert_knowledge_unit(
    conn: sqlite3.Connection,
    document_id: int,
    document_chunk_id: int,
    title: str,
    summary: str,
    keywords: list[str],
    embedding_text: str,
) -> int:
    cursor = conn.execute(
        """
            INSERT INTO document_chunk_knowledge_units (
                document_id,
                document_chunk_id,
                title,
                summary,
                keywords,
                embedding_text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
        (
            document_id,
            document_chunk_id,
            title,
            summary,
            json.dumps(keywords, ensure_ascii=False),
            embedding_text,
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("insert document_chunk_knowledge_units failed")

    return cursor.lastrowid


def query_by_ids(ids: list[int]) -> list[DocumentChunkKnowledgeUnits]:
    if not ids:
        return []
    with get_sqlite_connection() as conn:
        placeholders = ",".join(["?"] * len(ids))
        rows = conn.execute(
            f"""
            select * from {TABLE_NAME} where id in ({placeholders})
            """,
            ids,
        ).fetchall()

        result = []

        for row in rows:
            data = dict(row)
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            result.append(DocumentChunkKnowledgeUnits(**data))

        return result
