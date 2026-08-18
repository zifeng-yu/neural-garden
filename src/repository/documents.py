import sqlite3
from dataclasses import dataclass
from datetime import datetime

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO

TABLE_NAME = "documents"


@dataclass
class DocumentDO(BaseDO):
    # 文件名
    file_name: str
    # 文件名hash值
    file_name_hash: str
    # 文件内容hash值
    content_hash: str


def insert_document(
    conn: sqlite3.Connection, file_name: str, file_name_hash: str, content_hash: str
) -> int:
    if conn is not None:
        return _insert_document(conn, file_name, file_name_hash, content_hash)
    with get_sqlite_connection() as new_conn:
        return _insert_document(new_conn, file_name, file_name_hash, content_hash)


def _insert_document(
    conn: sqlite3.Connection, file_name: str, file_name_hash: str, content_hash: str
) -> int:
    cursor = conn.cursor()
    cursor.execute(
        f"""
        INSERT INTO {TABLE_NAME}
        (
            file_name,
            file_name_hash,
            content_hash
        )
        VALUES (?, ?, ?)
        """,
        (
            file_name,
            file_name_hash,
            content_hash,
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("insert document failed")
    return cursor.lastrowid


def query_by_file_name_hash(file_name_hash: str):
    with get_sqlite_connection() as conn:

        row = conn.execute(
            f"""
            select * from {TABLE_NAME} where file_name_hash = ?
            """,
            (file_name_hash,),
        ).fetchone()
        if row is None:
            return None

        data = dict(row)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return DocumentDO(**data)


def query_by_content_hash(content_hash: str):
    with get_sqlite_connection() as conn:

        row = conn.execute(
            f"""
                select * from {TABLE_NAME} where content_hash = ?
                """,
            (content_hash,),
        ).fetchone()
        if row is None:
            return None

        data = dict(row)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return DocumentDO(**data)


def query_by_id(id: int):
    with get_sqlite_connection() as conn:

        row = conn.execute(
            f"""
                select * from {TABLE_NAME} where id = ?
                """,
            (id,),
        ).fetchone()
        if row is None:
            return None

        data = dict(row)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return DocumentDO(**data)


def query_all() -> list[DocumentDO]:
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
            result.append(DocumentDO(**data))

        return result
