"""验证三个独立SQL查询与只读连接保证。"""

import sqlite3
from pathlib import Path

import pytest

import src.get_sqlite_connection as connections
from src.feedback_events_analytics import feedback_analytics as analytics
from src.get_sqlite_connection import get_sqlite_read_connection
from src.repository.create_table import create_tables


@pytest.fixture
def database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "feedback.sqlite"
    with sqlite3.connect(path) as conn:
        create_tables(conn)
        conn.executemany(
            """
            INSERT INTO search_sessions(session_id, query, result_count, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("s1", "RAG", 0, "2026-10-02 10:00:00"),
                ("s2", "RAG", 2, "2026-10-02 10:00:00"),
                ("s3", "Agent", 0, "2026-10-02 10:00:00"),
                ("old", "SQL", 0, "2026-08-01 10:00:00"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO feedback_events(session_id, event_type, document_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("s1", "click", 4, "2026-10-02 10:00:00"),
                ("s2", "click", 4, "2026-10-02 10:00:00"),
                ("s3", "click", None, "2026-10-02 10:00:00"),
                ("s1", "dwell", 4, "2026-10-02 10:00:00"),
                ("old", "click", 99, "2026-08-01 10:00:00"),
            ],
        )
    monkeypatch.setattr(connections, "SQLITE_TABLE", str(path))
    return path


def test_query_fields_date_filter_and_limit(database):
    assert analytics.get_popular_queries("2026-10-01", "2026-10-08") == [
        {"query": "RAG", "search_count": 2},
        {"query": "Agent", "search_count": 1},
    ]
    assert analytics.get_zero_result_queries("2026-10-01", "2026-10-08") == [
        {"query": "Agent", "search_count": 1},
        {"query": "RAG", "search_count": 1},
    ]
    assert analytics.get_popular_queries("2026-10-01", "2026-10-08", 1) == [
        {"query": "RAG", "search_count": 2}
    ]
    assert {
        row["query"]
        for row in analytics.get_popular_queries("2026-08-01", "2026-10-08")
    } == {
        "RAG",
        "Agent",
        "SQL",
    }


def test_document_statistics_exclude_null_group_and_count_only_clicks(database):
    assert analytics.get_popular_documents("2026-10-01", "2026-10-08") == [
        {"document_id": 4, "click_count": 2},
    ]
    assert analytics.get_popular_documents("2026-10-01", "2026-10-08", 1) == [
        {"document_id": 4, "click_count": 2}
    ]


def test_read_connection_refuses_writes_and_closes(database):
    with get_sqlite_read_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM search_sessions").fetchone()[0] == 4
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM search_sessions")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")


def test_read_connection_does_not_create_missing_parent(tmp_path, monkeypatch):
    path = tmp_path / "missing" / "feedback.sqlite"
    monkeypatch.setattr(connections, "SQLITE_TABLE", str(path))
    with pytest.raises(sqlite3.OperationalError):
        with get_sqlite_read_connection():
            pytest.fail("缺库时应在进入连接上下文前失败")
    assert not path.parent.exists()


def test_statistics_do_not_create_a_missing_database(tmp_path, monkeypatch):
    path = tmp_path / "feedback.sqlite"
    monkeypatch.setattr(connections, "SQLITE_TABLE", str(path))
    with pytest.raises(sqlite3.OperationalError):
        analytics.get_popular_queries("2026-10-01", "2026-10-08")
    assert not path.exists()
