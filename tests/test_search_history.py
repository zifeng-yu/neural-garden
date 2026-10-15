"""真实临时SQLite验证BM25/混合检索保存开关及事务，不访问外部服务。"""

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import chromadb
import pytest

import src.get_chroma_collection as chroma_storage
import src.get_sqlite_connection as connections
import src.rag_answer as rag
import src.retrieval.hybrid_search as hybrid
from src.repository import search_sessions
from src.repository.create_table import create_tables
from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum
from src.retrieval import es_search as bm25
from src.retrieval.search_result import SearchResult


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "history.sqlite"
    with sqlite3.connect(path) as conn:
        create_tables(conn)
    monkeypatch.setattr(connections, "SQLITE_TABLE", str(path))
    return path


def read_history(database):
    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        return (
            [dict(row) for row in conn.execute("SELECT * FROM search_sessions")],
            [
                dict(row)
                for row in conn.execute("SELECT * FROM search_results ORDER BY rank")
            ],
        )


@pytest.fixture
def es_client(monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    client.search.return_value = {
        "timed_out": False,
        "_shards": {"failed": 0},
        "hits": {
            "hits": [
                {
                    "_score": score,
                    "_source": {
                        "document_id": 7,
                        "document_chunk_id": chunk_id,
                        "retrieval_text": str(chunk_id),
                    },
                }
                for chunk_id, score in [(10, 3.5), (11, 2.0)]
            ]
        },
    }
    monkeypatch.setattr(bm25, "get_es_client", lambda: client)
    return client


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("empty", [True, False])
def test_bm25_saves_rank_scores_and_empty_session(database, es_client, enabled, empty):
    if empty:
        es_client.search.return_value["hits"]["hits"] = []
    hits = bm25.search_bm25(" HashSet ", save_session=enabled)
    sessions, results = read_history(database)
    if not enabled:
        assert sessions == results == []
        return
    assert len(sessions) == 1
    assert sessions[0]["query"] == "HashSet"
    assert sessions[0]["result_count"] == len(hits)
    assert [
        (row["document_chunk_id"], row["rank"], row["score"], row["raw_score"])
        for row in results
    ] == [(h.document_chunk_id, h.rank, h.score, h.score) for h in hits]
    assert all(
        row["retrieval_type"] == "KEYWORD/BM25"
        and row["source_type"] == "es_knowledge"
        and row["session_id"] == sessions[0]["session_id"]
        for row in results
    )


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("empty", [True, False])
def test_hybrid_only_saves_final_results(
    database, es_client, monkeypatch, enabled, empty
):
    if empty:
        es_client.search.return_value["hits"]["hits"] = []
    vector = (
        []
        if empty
        else [
            SearchResult(
                "摘要",
                7,
                11,
                0.9,
                1,
                SourceTypeEnum.CHROMA_KNOWLEDGE,
                RetrievalTypeEnum.VECTOR,
            )
        ]
    )
    vector_search = Mock(return_value=vector)
    monkeypatch.setattr(hybrid, "search_vector", vector_search)
    monkeypatch.setattr(
        hybrid, "get_existing_collection", lambda _: SimpleNamespace(count=lambda: 2)
    )
    hits = hybrid.hybrid_search("HashSet", top_k=1, save_session=enabled)
    vector_search.assert_called_once_with(
        query="HashSet", top_k=2, show_score=False, save_session=False
    )
    sessions, results = read_history(database)
    if not enabled:
        assert sessions == results == []
        return
    assert len(sessions) == 1
    assert sessions[0]["result_count"] == len(hits)
    assert len(results) == len(hits)
    assert [
        (row["document_chunk_id"], row["rank"], row["score"], row["raw_score"])
        for row in results
    ] == [(h.document_chunk_id, h.rank, h.score, h.score) for h in hits]
    assert all(
        row["retrieval_type"] == "hybrid" and row["source_type"] == "hybrid_knowledge"
        for row in results
    )


def test_failed_hybrid_does_not_save_bm25_candidates(database, es_client, monkeypatch):
    monkeypatch.setattr(
        hybrid, "search_vector", Mock(side_effect=RuntimeError("向量检索失败"))
    )
    monkeypatch.setattr(
        hybrid, "get_existing_collection", lambda _: SimpleNamespace(count=lambda: 2)
    )
    with pytest.raises(RuntimeError, match="向量检索失败"):
        hybrid.hybrid_search("HashSet")
    assert read_history(database) == ([], [])


@pytest.mark.parametrize("missing", ["directory", "collection"])
def test_hybrid_missing_chroma_raises_without_creating_or_saving(
    database, es_client, tmp_path, monkeypatch, missing
):
    directory = tmp_path / "chroma"
    client = (
        chromadb.PersistentClient(path=str(directory))
        if missing == "collection"
        else None
    )
    monkeypatch.setattr(chroma_storage, "PERSIST_DIRECTORY", str(directory))
    vector_search = Mock(side_effect=AssertionError("缺失源库不能执行向量检索"))
    monkeypatch.setattr(hybrid, "search_vector", vector_search)

    error = (
        FileNotFoundError if missing == "directory" else chromadb.errors.NotFoundError
    )
    with pytest.raises(error):
        hybrid.hybrid_search("HashSet")

    # BM25 已返回非空候选，也不能将缺失向量库伪装为成功的混合检索。
    es_client.search.assert_called_once()
    vector_search.assert_not_called()
    assert read_history(database) == ([], [])
    if client is None:
        assert not directory.exists()
    else:
        assert client.list_collections() == []


def test_hybrid_existing_empty_collection_keeps_bm25_results(
    database, es_client, tmp_path, monkeypatch
):
    directory = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(directory))
    collection = client.create_collection(hybrid.CHROMA_KNOWLEDGE_TABLE_NAME)
    monkeypatch.setattr(chroma_storage, "PERSIST_DIRECTORY", str(directory))
    vector_search = Mock(side_effect=AssertionError("空集合无需执行向量检索"))
    monkeypatch.setattr(hybrid, "search_vector", vector_search)

    hits = hybrid.hybrid_search("HashSet")

    assert [hit.document_chunk_id for hit in hits] == [10, 11]
    assert [hit.rank for hit in hits] == [1, 2]
    vector_search.assert_not_called()
    assert collection.count() == 0
    sessions, results = read_history(database)
    assert len(sessions) == 1
    assert sessions[0]["result_count"] == 2
    assert [row["document_chunk_id"] for row in results] == [10, 11]
    assert all(row["session_id"] == sessions[0]["session_id"] for row in results)


def test_history_write_failure_rolls_back_entire_session(
    database, es_client, monkeypatch
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
        bm25.search_bm25("HashSet")
    assert read_history(database) == ([], [])


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_rag_does_not_save_search_history(monkeypatch, mode):
    retrieve = Mock(return_value=[])
    monkeypatch.setattr(rag, "search", retrieve)
    result = rag.answer_question("HashSet", retrieval_mode=mode)
    assert result["sources"] == []
    assert retrieve.call_args.kwargs["save_session"] is False
    assert retrieve.call_args.kwargs["retrieval_mode"] == mode
