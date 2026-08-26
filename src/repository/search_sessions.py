import sqlite3
from dataclasses import dataclass

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO


@dataclass
class SearchSessionsDO(BaseDO):
    session_id: str
    query: str
    result_count: int


def _save_search_sessions(
    conn: sqlite3.Connection, session_id: str, query: str, result_count: int
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO search_sessions
        (
            session_id,
            query,
            result_count
        )
        VALUES (?, ?, ?)
        """,
        (
            session_id,
            query,
            result_count,
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("insert search_sessions failed")
    return cursor.lastrowid


def save_search_sessions(session_id: str, query: str, result_count: int) -> int:
    with get_sqlite_connection() as conn:
        return _save_search_sessions(conn, session_id, query, result_count)


def _update_result_count_by_session_id(
    conn: sqlite3.Connection, session_id: str, result_count: int
):
    conn.execute(
        """
        update search_sessions set result_count = ?  where session_id = ?
        """,
        (
            result_count,
            session_id,
        ),
    )


def update_result_count_by_session_id(session_id: str, result_count: int):
    with get_sqlite_connection() as conn:
        _update_result_count_by_session_id(conn, session_id, result_count)
