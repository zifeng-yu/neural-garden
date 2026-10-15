"""三个独立的使用与反馈SQL查询；日期校验和工具响应由调用方负责。"""

from typing import Any

from src.get_sqlite_connection import get_sqlite_read_connection
from src.repository.feedback_events import EventTypeEnum


def get_popular_queries(
    start_date: str, end_date_exclusive: str, limit: int = 5
) -> list[dict[str, Any]]:
    """查询日期区间内的热门搜索词，按记录次数降序排列。"""
    with get_sqlite_read_connection() as conn:
        rows = conn.execute(
            """
            SELECT query, COUNT(*) AS search_count
            FROM search_sessions
            WHERE created_at >= ? AND created_at < ?
            GROUP BY query
            ORDER BY search_count DESC, query ASC
            LIMIT ?
            """,
            (start_date + " 00:00:00", end_date_exclusive + " 00:00:00", limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_zero_result_queries(
    start_date: str, end_date_exclusive: str, limit: int = 5
) -> list[dict[str, Any]]:
    """查询结果数为零的搜索词，按记录次数降序排列。"""
    with get_sqlite_read_connection() as conn:
        rows = conn.execute(
            """
            SELECT query, COUNT(*) AS search_count
            FROM search_sessions
            WHERE created_at >= ? AND created_at < ? AND result_count = 0
            GROUP BY query
            ORDER BY search_count DESC, query ASC
            LIMIT ?
            """,
            (start_date + " 00:00:00", end_date_exclusive + " 00:00:00", limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_popular_documents(
    start_date: str, end_date_exclusive: str, limit: int = 5
) -> list[dict[str, Any]]:
    """查询文档点击事件次数；排除缺失文档ID及其他反馈类型。"""
    with get_sqlite_read_connection() as conn:
        rows = conn.execute(
            """
            SELECT document_id, COUNT(*) AS click_count
            FROM feedback_events
            WHERE created_at >= ? AND created_at < ?
              AND event_type = ? AND document_id IS NOT NULL
            GROUP BY document_id
            ORDER BY click_count DESC, document_id ASC
            LIMIT ?
            """,
            (
                start_date + " 00:00:00",
                end_date_exclusive + " 00:00:00",
                EventTypeEnum.CLICK.value,
                limit,
            ),
        ).fetchall()
    return [dict(row) for row in rows]
