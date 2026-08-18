import sqlite3
from dataclasses import dataclass
from datetime import datetime

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO

TABLE_NAME = "document_chunks"


@dataclass
class DocumentChunksDo(BaseDO):
    document_id: int
    split_no: int
    content: str
    content_hash: str


def insert_document_chunk(
    conn: sqlite3.Connection,
    document_id: int,
    split_no: int,
    content: str,
    content_hash: str,
) -> int:
    if conn is not None:
        return _insert_document_chunk(
            conn, document_id, split_no, content, content_hash
        )
    with get_sqlite_connection() as new_conn:
        return _insert_document_chunk(
            new_conn, document_id, split_no, content, content_hash
        )


def _insert_document_chunk(
    conn: sqlite3.Connection,
    document_id: int,
    split_no: int,
    content: str,
    content_hash: str,
) -> int:
    cursor = conn.execute(
        """
            INSERT INTO document_chunks (
                document_id,
                split_no,
                content,
                content_hash
            )
            VALUES (?, ?, ?, ?)
            """,
        (
            document_id,
            split_no,
            content,
            content_hash,
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("insert document_chunks failed")
    return cursor.lastrowid


def query_by_ids(ids: list[int]) -> list[DocumentChunksDo]:
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
            result.append(DocumentChunksDo(**data))

        return result
