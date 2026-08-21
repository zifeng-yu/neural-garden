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
    conn: sqlite3.Connection | None,
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


def query_by_document_id(document_id: int) -> list[DocumentChunkKnowledgeUnits]:
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
            result.append(DocumentChunkKnowledgeUnits(**data))

        return result


def _query_by_document_id(
    conn: sqlite3.Connection, document_id: int
) -> list[DocumentChunkKnowledgeUnits]:
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
        result.append(DocumentChunkKnowledgeUnits(**data))

    return result


def copy_documents_chunk_knowledge_unit(
    conn: sqlite3.Connection,
    need_document_id: int,
    new_document_id: int,
    old_new_chunk_id_map: dict[int, int],
) -> list[int]:
    knowledge_unit_list = _query_by_document_id(conn, need_document_id)
    if not knowledge_unit_list:
        return []
    new_knowledge_unit_ids = []
    for knowledge_unit in knowledge_unit_list:
        new_knowledge_unit_id = insert_knowledge_unit(
            conn,
            new_document_id,
            old_new_chunk_id_map[knowledge_unit.document_chunk_id],
            knowledge_unit.title,
            knowledge_unit.summary,
            knowledge_unit.keywords,
            knowledge_unit.embedding_text,
        )
        new_knowledge_unit_ids.append(new_knowledge_unit_id)
    return new_knowledge_unit_ids


def delete_by_document_id(document_id: int):
    with get_sqlite_connection() as conn:
        conn.execute(
            f"""
                    delete from {TABLE_NAME} where document_id = ?
                """,
            (document_id,),
        )
