"""向量检索的返回契约和SQLite事务；外部模型与Chroma使用替身。"""

import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.get_sqlite_connection as connections
import src.retrieval.chroma_search as search_module
from src.repository import search_sessions
from src.repository.create_table import create_tables
from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum


@pytest.fixture
def search_dependencies(tmp_path, monkeypatch):
    database = tmp_path / "search.sqlite"
    with sqlite3.connect(database) as conn:
        create_tables(conn)
    monkeypatch.setattr(connections, "SQLITE_TABLE", str(database))
    collection = Mock()
    collection.count.return_value = 2
    collection.query.return_value = {
        "ids": [["knowledge-1", "knowledge-2"]],
        "documents": [["第一条知识", "第二条知识"]],
        "metadatas": [
            [
                {"document_id": 11, "document_chunk_id": 101},
                {"document_id": 22, "document_chunk_id": 202},
            ]
        ],
        "distances": [[0.1, 0.3]],
    }
    get_embedding = Mock(return_value=[0.1, 0.2, 0.3])
    get_existing_collection = Mock(return_value=collection)
    monkeypatch.setattr(search_module, "get_embedding", get_embedding)
    monkeypatch.setattr(
        search_module, "get_existing_collection", get_existing_collection
    )
    monkeypatch.setattr(search_module, "log_results", Mock())
    return SimpleNamespace(
        database=database,
        collection=collection,
        get_embedding=get_embedding,
        get_existing_collection=get_existing_collection,
    )


def read_history(dependencies):
    with sqlite3.connect(dependencies.database) as conn:
        conn.row_factory = sqlite3.Row
        return (
            [dict(row) for row in conn.execute("SELECT * FROM search_sessions")],
            [
                dict(row)
                for row in conn.execute("SELECT * FROM search_results ORDER BY rank")
            ],
        )


def test_search_returns_ranked_results(search_dependencies):
    results = search_module.search_vector(
        "什么是负利率", top_k=2, show_score=False, save_session=False
    )
    assert len(results) == 2
    first = results[0]
    assert first.content == "第一条知识"
    assert first.document_id == 11
    assert first.document_chunk_id == 101
    assert first.score == pytest.approx(0.9)
    assert first.raw_score == 0.1
    assert first.rank == 1
    assert first.source_type == SourceTypeEnum.CHROMA_KNOWLEDGE.value
    assert first.retrieval_type == RetrievalTypeEnum.VECTOR.value
    assert results[1].score == pytest.approx(0.7)
    assert results[1].rank == 2
    search_dependencies.collection.query.assert_called_once_with(
        query_embeddings=[0.1, 0.2, 0.3],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )


def test_search_does_not_open_database_when_save_session_is_false(
    search_dependencies, monkeypatch
):
    monkeypatch.setattr(
        connections,
        "SQLITE_TABLE",
        str(search_dependencies.database.parent / "missing" / "db.sqlite"),
    )
    assert len(search_module.search_vector("测试查询", save_session=False)) == 2
    assert read_history(search_dependencies) == ([], [])
    assert not (search_dependencies.database.parent / "missing").exists()


def test_search_persists_session_and_results(search_dependencies):
    results = search_module.search_vector("测试查询")
    sessions, rows = read_history(search_dependencies)
    assert len(sessions) == 1
    assert sessions[0]["query"] == "测试查询"
    assert sessions[0]["result_count"] == 2
    assert [row["document_chunk_id"] for row in rows] == [101, 202]
    assert [row["rank"] for row in rows] == [1, 2]
    assert [row["score"] for row in rows] == pytest.approx([r.score for r in results])
    # 原始余弦距离保持不变，不能写成向量相似度。
    assert [row["raw_score"] for row in rows] == [0.1, 0.3]
    assert all(
        row["session_id"] == sessions[0]["session_id"]
        and row["retrieval_type"] == "vector"
        and row["source_type"] == "chroma_knowledge"
        for row in rows
    )


def test_search_raises_when_embedding_fails(search_dependencies):
    search_dependencies.get_embedding.return_value = None
    with pytest.raises(RuntimeError, match="未能生成查询 embedding"):
        search_module.search_vector("测试查询")
    search_dependencies.get_existing_collection.assert_not_called()
    assert read_history(search_dependencies) == ([], [])


@pytest.mark.parametrize("empty_collection", [True, False])
def test_empty_search_still_saves_zero_result_session(
    search_dependencies, empty_collection
):
    if empty_collection:
        search_dependencies.collection.count.return_value = 0
    else:
        search_dependencies.collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
    assert search_module.search_vector("没有结果的查询") == []
    sessions, rows = read_history(search_dependencies)
    assert len(sessions) == 1 and sessions[0]["result_count"] == 0
    assert rows == []
    if empty_collection:
        search_dependencies.collection.query.assert_not_called()


def test_failed_chroma_query_does_not_save_session(search_dependencies):
    search_dependencies.collection.query.side_effect = RuntimeError("Chroma连接失败")
    with pytest.raises(RuntimeError, match="Chroma连接失败"):
        search_module.search_vector("测试查询")
    assert read_history(search_dependencies) == ([], [])


def test_result_write_failure_rolls_back_session_and_previous_result(
    search_dependencies, monkeypatch
):
    original = search_sessions._save_search_results
    count = 0

    def fail_second(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("模拟保存失败")
        return original(*args)

    monkeypatch.setattr(search_sessions, "_save_search_results", fail_second)
    with pytest.raises(RuntimeError, match="模拟保存失败"):
        search_module.search_vector("测试查询")
    assert read_history(search_dependencies) == ([], [])
