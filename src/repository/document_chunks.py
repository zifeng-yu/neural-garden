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
    conn: sqlite3.Connection | None,
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


def query_by_document_id(document_id: int) -> list[DocumentChunksDo]:
    with get_sqlite_connection() as conn:
        rows = conn.execute(
            f"""
            select * from {TABLE_NAME} where document_id = ?
            """,
            (document_id,),
        ).fetchall()

        result = []

        for row in rows:
            data = dict(row)
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            result.append(DocumentChunksDo(**data))

        return result


def _query_by_document_id(
    conn: sqlite3.Connection, document_id: int
) -> list[DocumentChunksDo]:
    rows = conn.execute(
        f"""
        select * from {TABLE_NAME} where document_id = ?
        """,
        (document_id,),
    ).fetchall()

    result = []

    for row in rows:
        data = dict(row)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        result.append(DocumentChunksDo(**data))

    return result


def copy_document_chunks(
    conn: sqlite3.Connection, need_document_id: int, new_document_id: int
) -> dict[int, int]:
    chunk_list = _query_by_document_id(conn, need_document_id)
    if not chunk_list:
        return {}
    chunk_id_map = {}
    for chunk in chunk_list:
        new_chunk_id = insert_document_chunk(
            conn, new_document_id, chunk.split_no, chunk.content, chunk.content_hash
        )
        chunk_id_map[chunk.id] = new_chunk_id
    return chunk_id_map


def delete_by_document_id(conn: sqlite3.Connection, document_id: int):
    conn.execute(
        f"""
                delete from {TABLE_NAME} where document_id = ?
            """,
        (document_id,),
    )
