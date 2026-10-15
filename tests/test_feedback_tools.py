"""三个指标工具的契约、日期边界和数据库异常离线测试。"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

import src.get_sqlite_connection as connections
from src.repository.create_table import create_tables
from src.tools.feedback_tools import UsageSummaryInput, create_usage_tools

TOOLS = {tool.name.removeprefix("query_"): tool for tool in create_usage_tools()}

SESSIONS = [
    ("s0", "RAG", 0, "2026-09-30 23:59:59"),
    ("s1", "RAG", 2, "2026-10-01 00:00:00"),
    ("s2", "RAG", 0, "2026-10-02 10:00:00"),
    ("s3", "Agent", 0, "2026-10-03 10:00:00"),
    ("s4", "ES", 0, "2026-10-04 10:00:00"),
    ("s5", "ES", 1, "2026-10-05 10:00:00"),
    ("s6", "RAG", 0, "2026-10-07 23:59:59"),
    ("s7", "Python", 1, "2026-10-06 10:00:00"),
    ("s8", "SQL", 1, "2026-10-06 11:00:00"),
    ("s9", "Z", 1, "2026-10-06 12:00:00"),
    ("s10", "RAG", 0, "2026-10-08 00:00:00"),
]
EVENTS = [
    ("s1", "click", 4, "2026-10-01 00:00:00"),
    ("s1", "click", 4, "2026-10-01 00:01:00"),
    ("s0", "click", 4, "2026-10-02 10:00:00"),
    ("s3", "click", 6, "2026-10-03 10:00:00"),
    ("s4", "click", 6, "2026-10-04 10:00:00"),
    ("s5", "click", 1, "2026-10-05 10:00:00"),
    ("s7", "click", 2, "2026-10-06 10:00:00"),
    ("s8", "click", 3, "2026-10-06 11:00:00"),
    ("s9", "click", 9, "2026-10-07 23:59:59"),
    ("s3", "dwell", 6, "2026-10-03 10:01:00"),
    ("s3", "click", None, "2026-10-03 10:02:00"),
    ("s0", "click", 99, "2026-09-30 23:59:59"),
    ("s10", "click", 99, "2026-10-08 00:00:00"),
]

EXPECTED = {
    "popular_queries": [("RAG", 3), ("ES", 2), ("Agent", 1), ("Python", 1), ("SQL", 1)],
    "zero_result_queries": [("RAG", 2), ("Agent", 1), ("ES", 1)],
    "popular_documents": [(4, 3), (6, 2), (1, 1), (2, 1), (3, 1)],
}


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "fixture.sqlite"
    # 只在测试临时目录创建新数据库，不覆盖已有文件。
    with path.open("xb"):
        pass
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        create_tables(conn)
        conn.executemany(
            """
            INSERT INTO search_sessions(session_id, query, result_count, created_at)
            VALUES (?, ?, ?, ?)
            """,
            SESSIONS,
        )
        conn.executemany(
            """
            INSERT INTO feedback_events(session_id, event_type, document_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            EVENTS,
        )
        conn.commit()
    monkeypatch.setattr(connections, "SQLITE_TABLE", str(path))
    return path


@pytest.mark.parametrize("metric", list(EXPECTED))
def test_golden(db, metric):
    result = TOOLS[metric].invoke(
        {"start_date": "2026-10-01", "end_date_exclusive": "2026-10-08"}
    )
    assert result == {
        "status": "ok",
        "metric": metric,
        "start_date": "2026-10-01",
        "end_date_exclusive": "2026-10-08",
        "timezone": "Asia/Shanghai",
        "limit": 5,
        "items": [{"key": key, "count": count} for key, count in EXPECTED[metric]],
    }


@pytest.mark.parametrize("metric", list(EXPECTED))
def test_empty_is_success(db, metric):
    r = TOOLS[metric].invoke(
        {"start_date": "2026-11-01", "end_date_exclusive": "2026-11-02"}
    )
    assert r["status"] == "ok" and r["items"] == []


def test_oct8_is_not_oct7(db):
    r = TOOLS["popular_documents"].invoke(
        {"start_date": "2026-10-08", "end_date_exclusive": "2026-10-09"}
    )
    assert r["items"] == [{"key": 99, "count": 1}]


@pytest.mark.parametrize("metric", list(EXPECTED))
@pytest.mark.parametrize(
    "start,end",
    [
        ("20261001", "2026-10-08"),
        ("2026-02-30", "2026-10-08"),
        ("2026-10-08", "2026-10-08"),
        ("2026-10-09", "2026-10-08"),
    ],
)
def test_invalid_inputs(db, metric, start, end):
    with pytest.raises(ValidationError):
        TOOLS[metric].invoke({"start_date": start, "end_date_exclusive": end})


def test_missing_database_is_not_empty(tmp_path, monkeypatch):
    path = tmp_path / "missing.sqlite"
    monkeypatch.setattr(connections, "SQLITE_TABLE", str(path))
    with pytest.raises(sqlite3.OperationalError):
        TOOLS["popular_queries"].invoke(
            {"start_date": "2026-10-01", "end_date_exclusive": "2026-10-08"}
        )
    assert not path.exists()


def test_missing_tables_is_not_empty(tmp_path, monkeypatch):
    path = tmp_path / "blank.sqlite"
    sqlite3.connect(path).close()
    monkeypatch.setattr(connections, "SQLITE_TABLE", str(path))
    with pytest.raises(sqlite3.OperationalError):
        TOOLS["popular_queries"].invoke(
            {"start_date": "2026-10-01", "end_date_exclusive": "2026-10-08"}
        )


def test_three_tool_schemas_and_real_execution(db):
    tools = create_usage_tools()
    assert [tool.name for tool in tools] == [
        "query_popular_queries",
        "query_zero_result_queries",
        "query_popular_documents",
    ]
    args = {"start_date": "2026-10-01", "end_date_exclusive": "2026-10-08"}
    for tool in tools:
        schema = tool.get_input_schema().model_json_schema()
        assert set(schema["properties"]) == {"start_date", "end_date_exclusive"}
        assert schema.get("additionalProperties") is False
        with pytest.raises(ValidationError):
            tool.invoke({**args, "metric": "popular_queries"})
        with pytest.raises(ValidationError):
            tool.invoke({**args, "sql": "SELECT 1"})
    assert TOOLS["zero_result_queries"].invoke(args)["items"] == [
        {"key": "RAG", "count": 2},
        {"key": "Agent", "count": 1},
        {"key": "ES", "count": 1},
    ]
    with pytest.raises(ValidationError):
        UsageSummaryInput(**args, sql="SELECT 1")
