import sqlite3
import uuid
from dataclasses import dataclass

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO
from src.repository.search_results import (
    RetrievalTypeEnum,
    SourceTypeEnum,
    _save_search_results,
)
from src.retrieval.search_result import SearchResult


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


def save_search_session_with_results(
    query: str,
    results: list[SearchResult],
    retrieval_type: RetrievalTypeEnum,
    source_type: SourceTypeEnum,
) -> str:
    """同一事务保存session与最终结果；空结果保留session，失败整体回滚。

    向量结果保留原始距离；未提供raw_score时使用排序分数（BM25/RRF）。
    """
    session_id = uuid.uuid4().hex
    with get_sqlite_connection() as conn:
        _save_search_sessions(conn, session_id, query, len(results))
        for result in results:
            _save_search_results(
                conn,
                session_id,
                result.document_id,
                result.document_chunk_id,
                result.rank,
                result.score,
                result.raw_score if result.raw_score is not None else result.score,
                retrieval_type,
                source_type,
            )
    return session_id
