"""核对初始化的保留数据行为，以及显式重置入口；只使用临时库。"""

import sys
from unittest.mock import Mock

import pytest

import src.get_sqlite_connection as connections
import src.indexer as indexer
import src.repository.init as initialization


@pytest.fixture
def sqlite_path(tmp_path, monkeypatch):
    path = tmp_path / "sqlite" / "neural.db"
    monkeypatch.setattr(connections, "SQLITE_TABLE", str(path))
    monkeypatch.setattr(initialization, "setup_logging", Mock())
    return path


def test_default_initialization_creates_database(sqlite_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["src.repository.init"])
    initialization.main()

    assert sqlite_path.is_file()
    with connections.get_sqlite_read_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    assert len(tables) == 10


def test_default_initialization_preserves_data_and_restores_missing_table(
    sqlite_path, monkeypatch
):
    initialization.sqlite_table_create_if_not()
    with connections.get_sqlite_connection() as conn:
        conn.execute(
            "INSERT INTO documents(file_name,file_name_hash,content_hash) "
            "VALUES(?,?,?)",
            ("keep.md", "name-hash", "content-hash"),
        )
        conn.execute("DROP TABLE insights")

    monkeypatch.setattr(sys, "argv", ["src.repository.init"])
    initialization.main()
    initialization.main()

    with connections.get_sqlite_read_connection() as conn:
        assert (
            conn.execute("SELECT file_name FROM documents").fetchone()[0] == "keep.md"
        )
        assert conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0] == 0


def test_explicit_sqlite_reset_removes_data_and_recreates_tables(
    sqlite_path, monkeypatch
):
    initialization.sqlite_table_create_if_not()
    with connections.get_sqlite_connection() as conn:
        conn.execute(
            "INSERT INTO documents(file_name,file_name_hash,content_hash) "
            "VALUES(?,?,?)",
            ("old.md", "name-hash", "content-hash"),
        )

    monkeypatch.setattr(sys, "argv", ["src.repository.init", "--reset"])
    initialization.main()

    with connections.get_sqlite_read_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0] == 0


@pytest.mark.parametrize("reset", [False, True])
def test_indexer_requires_explicit_flag_for_sqlite_and_chroma_reset(monkeypatch, reset):
    mocks = {}
    for name in (
        "setup_logging",
        "sqlite_table_create_if_not",
        "sqlite_table_reset",
        "resetDB_KNOWLEDGE",
        "resetDB_CONCEPT",
        "resetDB_INSIGHT",
        "index_directory",
    ):
        mocks[name] = Mock()
        monkeypatch.setattr(indexer, name, mocks[name])
    monkeypatch.setattr(indexer, "PILOT_DATASET_PATH", "/configured/pilot")
    monkeypatch.setattr(
        sys, "argv", ["src.indexer"] + (["--resetAllDB"] if reset else [])
    )

    indexer.main()

    mocks["index_directory"].assert_called_once_with("/configured/pilot")
    if reset:
        mocks["sqlite_table_create_if_not"].assert_not_called()
    else:
        mocks["sqlite_table_create_if_not"].assert_called_once_with()
    for name in (
        "sqlite_table_reset",
        "resetDB_KNOWLEDGE",
        "resetDB_CONCEPT",
        "resetDB_INSIGHT",
    ):
        if reset:
            mocks[name].assert_called_once_with()
        else:
            mocks[name].assert_not_called()
