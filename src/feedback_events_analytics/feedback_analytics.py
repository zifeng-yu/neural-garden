import logging

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.feedback_events import EventTypeEnum

logger = logging.getLogger(__name__)


def get_popular_queries(days: int = 7, limit: int = 10):
    """热门搜索"""
    with get_sqlite_connection() as conn:
        rows = conn.execute(
            """
            select query,count(*) AS search_count
            from search_sessions where created_at >= datetime('now', '+8 hours', ?)
            group by query order by search_count desc limit ?
            """,
            (
                f"-{days} days",
                limit,
            ),
        ).fetchall()
        return [dict(row) for row in rows]


def get_popular_documents(days: int = 7, limit: int = 10):
    """热门文档"""
    with get_sqlite_connection() as conn:
        rows = conn.execute(
            """
            select document_id,count(*) AS click_count
            from feedback_events where created_at >= datetime('now', '+8 hours', ?)
            and event_type = ?
            group by document_id order by click_count desc limit ?
            """,
            (
                f"-{days} days",
                EventTypeEnum.CLICK.value,
                limit,
            ),
        ).fetchall()
        return [dict(row) for row in rows]


def get_zero_result_queries(days: int = 7, limit: int = 10):
    """查询无结果搜索"""
    with get_sqlite_connection() as conn:
        rows = conn.execute(
            """
            select query,count(*) AS search_count
            from search_sessions where created_at >= datetime('now', '+8 hours', ?)
            and result_count = ?
            group by query order by search_count desc limit ?
            """,
            (
                f"-{days} days",
                0,
                limit,
            ),
        ).fetchall()
        return [dict(row) for row in rows]


if __name__ == "__main__":
    logger.info(get_popular_queries())
    logger.info(get_popular_documents())
    logger.info(get_zero_result_queries())
