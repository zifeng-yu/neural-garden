from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.search as search_module
from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum


@pytest.fixture
def search_dependencies(monkeypatch):
    """隔离 Embedding、Chroma 和 SQLite 等外部依赖。"""

    chroma_results = {
        "ids": [["knowledge-1", "knowledge-2"]],
        "documents": [["第一条知识", "第二条知识"]],
        "metadatas": [
            [
                {
                    "document_id": 11,
                    "document_chunk_id": 101,
                },
                {
                    "document_id": 22,
                    "document_chunk_id": 202,
                },
            ]
        ],
        "distances": [[0.1, 0.3]],
    }

    collection = Mock()
    collection.count.return_value = 2
    collection.query.return_value = chroma_results

    get_embedding = Mock(return_value=[0.1, 0.2, 0.3])
    get_collection = Mock(return_value=collection)
    save_search_sessions = Mock()
    save_search_results = Mock()
    update_result_count = Mock()
    log_results = Mock()

    monkeypatch.setattr(
        search_module,
        "get_embedding",
        get_embedding,
    )
    monkeypatch.setattr(
        search_module,
        "get_collection",
        get_collection,
    )
    monkeypatch.setattr(
        search_module,
        "save_search_sessions",
        save_search_sessions,
    )
    monkeypatch.setattr(
        search_module,
        "save_search_results",
        save_search_results,
    )
    monkeypatch.setattr(
        search_module,
        "update_result_count_by_session_id",
        update_result_count,
    )
    monkeypatch.setattr(
        search_module,
        "log_results",
        log_results,
    )

    # 固定 session_id，避免测试结果随机。
    monkeypatch.setattr(
        search_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="test-session-id"),
    )

    return SimpleNamespace(
        collection=collection,
        get_embedding=get_embedding,
        get_collection=get_collection,
        save_session=save_search_sessions,
        save_result=save_search_results,
        update_count=update_result_count,
    )


def test_search_returns_ranked_results(search_dependencies):
    results = search_module.search(
        "什么是负利率",
        top_k=2,
        show_score=False,
        save_session=False,
    )
    assert results is not None
    assert len(results) == 2
    first = results[0]
    assert first.content == "第一条知识"
    assert first.document_id == 11
    assert first.document_chunk_id == 101
    assert first.score == pytest.approx(0.9)
    assert first.rank == 1
    assert first.source_type == SourceTypeEnum.CHROMA_KNOWLEDGE.value
    assert first.retrieval_type == RetrievalTypeEnum.VECTOR.value

    second = results[1]
    assert second.score == pytest.approx(0.7)
    assert second.rank == 2

    search_dependencies.collection.query.assert_called_once_with(
        query_embeddings=[0.1, 0.2, 0.3],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )


def test_search_does_not_persist_when_save_session_is_false(
    search_dependencies,
):
    results = search_module.search(
        "测试查询",
        save_session=False,
    )

    assert results is not None
    search_dependencies.save_session.assert_not_called()
    search_dependencies.save_result.assert_not_called()
    search_dependencies.update_count.assert_not_called()


def test_search_persists_session_and_results(search_dependencies):
    results = search_module.search(
        "测试查询",
        save_session=True,
    )

    assert results is not None

    search_dependencies.save_session.assert_called_once_with(
        "test-session-id",
        "测试查询",
        0,
    )

    assert search_dependencies.save_result.call_count == 2

    first_call = search_dependencies.save_result.call_args_list[0].args

    assert first_call[0] == "test-session-id"
    assert first_call[1] == 11
    assert first_call[2] == 101
    assert first_call[3] == 1
    assert first_call[4] == pytest.approx(0.9)
    assert first_call[5] == pytest.approx(0.1)
    assert first_call[6] is RetrievalTypeEnum.VECTOR
    assert first_call[7] is SourceTypeEnum.CHROMA_KNOWLEDGE

    search_dependencies.update_count.assert_called_once_with(
        "test-session-id",
        2,
    )


def test_search_returns_none_when_embedding_fails(search_dependencies):
    search_dependencies.get_embedding.return_value = None

    results = search_module.search(
        "测试查询",
        save_session=True,
    )

    assert results is None
    search_dependencies.get_collection.assert_not_called()
    search_dependencies.save_session.assert_not_called()
    search_dependencies.save_result.assert_not_called()
    search_dependencies.update_count.assert_not_called()


def test_search_returns_empty_list_for_empty_collection(
    search_dependencies,
):
    search_dependencies.collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    results = search_module.search(
        "没有结果的查询",
        save_session=True,
    )

    assert results == []
    search_dependencies.save_result.assert_not_called()
    search_dependencies.update_count.assert_called_once_with(
        "test-session-id",
        0,
    )
