"""ES同步契约与源数据校验；源库使用临时SQLite/Chroma，ES使用替身。"""

import copy
import sqlite3
from unittest.mock import Mock

import chromadb
import pytest
from elasticsearch.helpers import expand_action

import src.get_chroma_collection as chroma_storage
import src.get_sqlite_connection as sqlite_storage
from src.es import client as es_client
from src.es import index as es_index
from src.es import ingest
from src.es.index import INDEX_MAPPING, create_index
from src.repository.create_table import create_tables
from src.retrieval import es_search as search


@pytest.fixture(autouse=True)
def configured_es(monkeypatch):
    for module in (ingest, search, es_index):
        monkeypatch.setattr(module, "ES_INDEX_NAME", "test-ku")
    monkeypatch.setattr(es_client, "ES_URL", "http://127.0.0.1:19200")


@pytest.fixture
def sources(tmp_path, monkeypatch):
    database = tmp_path / "source.sqlite"
    with sqlite3.connect(database) as conn:
        create_tables(conn)
        conn.execute(
            """
            INSERT INTO documents(id, file_name, file_name_hash, content_hash)
            VALUES (1, 'one.md', 'file-one', 'content-one')
            """
        )
        for unit_id, chunk_id, text in [(1, 10, "集合HashSet"), (2, 11, "狐狸型专家")]:
            conn.execute(
                """
                INSERT INTO document_chunks
                    (id, document_id, split_no, content, content_hash)
                VALUES (?, 1, ?, ?, ?)
                """,
                (chunk_id, unit_id, f"原文{text}", f"hash-{chunk_id}"),
            )
            conn.execute(
                """
                INSERT INTO document_chunk_knowledge_units
                    (id, document_id, document_chunk_id, title, summary,
                     keywords, embedding_text)
                VALUES (?, 1, ?, ?, ?, '[]', ?)
                """,
                (unit_id, chunk_id, text, text, text),
            )
    directory = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(directory))
    collection = client.create_collection("knowledge")
    collection.upsert(
        ids=["1", "2"],
        documents=["集合HashSet", "狐狸型专家"],
        embeddings=[[0.1, 0.2], [0.2, 0.1]],
        metadatas=[
            {"document_id": 1, "document_chunk_id": 10},
            {"document_id": 1, "document_chunk_id": 11},
        ],
    )
    monkeypatch.setattr(chroma_storage, "PERSIST_DIRECTORY", str(directory))
    monkeypatch.setattr(sqlite_storage, "SQLITE_TABLE", str(database))
    return database, collection


def test_export_joins_original_hash_and_chroma_identity(sources):
    assert ingest.export_data() == [
        {
            "knowledge_unit_id": unit_id,
            "document_id": 1,
            "document_chunk_id": chunk_id,
            "retrieval_text": text,
            "chunk_hash": f"hash-{chunk_id}",
        }
        for unit_id, chunk_id, text in [(1, 10, "集合HashSet"), (2, 11, "狐狸型专家")]
    ]


@pytest.mark.parametrize("mismatch", ["missing", "extra", "text", "metadata", "orphan"])
def test_mismatched_sources_cannot_reset_es(sources, monkeypatch, mismatch):
    database, collection = sources
    if mismatch == "missing":
        collection.delete(ids=["1"])
    elif mismatch == "extra":
        collection.upsert(
            ids=["3"],
            documents=["多余"],
            embeddings=[[0.3, 0.1]],
            metadatas=[{"document_id": 1, "document_chunk_id": 12}],
        )
    elif mismatch == "text":
        collection.update(ids=["1"], documents=["变更文本"], embeddings=[[0.4, 0.1]])
    elif mismatch == "metadata":
        collection.update(ids=["1"], metadatas=[{"document_id": 99}])
    else:
        with sqlite3.connect(database) as conn:
            conn.execute("DELETE FROM document_chunks WHERE id = 10")
    connect = Mock(side_effect=AssertionError("源数据异常不能访问或重置ES"))
    monkeypatch.setattr(ingest, "get_es_client", connect)
    with pytest.raises(ValueError, match="核对失败"):
        ingest.sync_index("reset")
    connect.assert_not_called()


def test_missing_source_collection_is_not_created(tmp_path, monkeypatch):
    directory = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(directory))
    monkeypatch.setattr(chroma_storage, "PERSIST_DIRECTORY", str(directory))
    with pytest.raises(chromadb.errors.NotFoundError):
        chroma_storage.get_existing_collection("knowledge")
    assert client.list_collections() == []


def test_missing_source_directory_is_not_created(tmp_path, monkeypatch):
    directory = tmp_path / "missing-chroma"
    monkeypatch.setattr(chroma_storage, "PERSIST_DIRECTORY", str(directory))
    with pytest.raises(FileNotFoundError):
        chroma_storage.get_existing_collection("knowledge")
    assert not directory.exists()


def test_skip_does_not_open_sources_or_es(monkeypatch):
    export = Mock(side_effect=AssertionError("skip不能导出"))
    connect = Mock(side_effect=AssertionError("skip不能连接ES"))
    monkeypatch.setattr(ingest, "export_data", export)
    monkeypatch.setattr(ingest, "get_es_client", connect)
    assert ingest.sync_index("skip") == {
        "mode": "skip",
        "index_name": "test-ku",
        "processed": 0,
    }
    export.assert_not_called()
    connect.assert_not_called()


def test_empty_source_upsert_preserves_existing_documents(monkeypatch):
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.count.return_value = {"count": 3}
    monkeypatch.setattr(ingest, "export_data", lambda: [])
    monkeypatch.setattr(ingest, "get_es_client", lambda: client)
    monkeypatch.setattr(ingest, "create_index", Mock(return_value=False))
    write = Mock(side_effect=AssertionError("空源库不能调用bulk或隐式清空旧索引"))
    monkeypatch.setattr(ingest, "upsert_documents", write)
    result = ingest.sync_index("upsert")
    assert result["processed"] == 0
    assert result["document_count"] == 3
    client.indices.delete.assert_not_called()
    write.assert_not_called()


def test_create_index_preserves_existing_and_reset_is_explicit():
    client = Mock()
    client.indices.exists.return_value = True
    client.indices.get_mapping.return_value = {
        "test-ku": {"mappings": copy.deepcopy(INDEX_MAPPING)}
    }
    assert create_index(client) is False
    client.indices.delete.assert_not_called()
    client.indices.create.assert_not_called()
    client.indices.delete.return_value = {"acknowledged": True}
    client.indices.create.return_value = {"acknowledged": True}
    assert create_index(client, reset=True) is True
    client.indices.delete.assert_called_once_with(index="test-ku")
    assert client.indices.create.call_args.kwargs["mappings"] == INDEX_MAPPING


def test_missing_index_is_created_without_deletion():
    client = Mock()
    client.indices.exists.return_value = False
    client.indices.create.return_value = {"acknowledged": True}
    assert create_index(client) is True
    client.indices.delete.assert_not_called()


def test_incompatible_mapping_is_not_silently_updated():
    client = Mock()
    client.indices.exists.return_value = True
    mapping = copy.deepcopy(INDEX_MAPPING)
    mapping["properties"]["retrieval_text"]["analyzer"] = "standard"
    client.indices.get_mapping.return_value = {"test-ku": {"mappings": mapping}}
    with pytest.raises(ValueError, match="不兼容"):
        create_index(client)
    client.indices.delete.assert_not_called()
    client.indices.create.assert_not_called()


@pytest.mark.parametrize("name", ["*", "_all", "test-ku,other", "Test", ""])
def test_reset_rejects_ambiguous_index_targets(name, monkeypatch):
    monkeypatch.setattr(es_index, "ES_INDEX_NAME", name)
    client = Mock()
    with pytest.raises(ValueError):
        create_index(client, reset=True)
    client.indices.exists.assert_not_called()


def test_repeated_upsert_keeps_ids_and_uses_sdk_update_payload(sources, monkeypatch):
    records = ingest.export_data()
    payloads = []

    def capture_bulk(client, actions, **kwargs):
        payloads.append([expand_action(action) for action in actions])
        assert kwargs["refresh"] == "wait_for"
        return len(payloads[-1]), []

    monkeypatch.setattr(ingest, "bulk", capture_bulk)
    assert ingest.upsert_documents(Mock(), records) == 2
    records[0]["retrieval_text"] = "更新后的文本"
    assert ingest.upsert_documents(Mock(), records) == 2
    assert [item[0] for item in payloads[0]] == [item[0] for item in payloads[1]]
    assert payloads[1][0] == (
        {"update": {"_index": "test-ku", "_id": "1"}},
        {"doc": {**records[0], "knowledge_unit_id": "1"}, "doc_as_upsert": True},
    )


@pytest.mark.parametrize("mode", ["upsert", "reset"])
def test_sync_creates_then_upserts_and_reports_count(sources, monkeypatch, mode):
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.count.return_value = {"count": 2}
    monkeypatch.setattr(ingest, "get_es_client", lambda: client)
    create = Mock(return_value=mode == "reset")
    write = Mock(return_value=2)
    monkeypatch.setattr(ingest, "create_index", create)
    monkeypatch.setattr(ingest, "upsert_documents", write)
    result = ingest.sync_index(mode)
    create.assert_called_once_with(client, reset=mode == "reset")
    assert client.count.call_args.kwargs["index"] == "test-ku"
    assert result["index_name"] == "test-ku"
    assert write.call_args.args[1][0]["knowledge_unit_id"] == 1
    assert result["processed"] == result["document_count"] == 2


def test_bulk_failure_is_not_reported_as_success(sources, monkeypatch):
    monkeypatch.setattr(ingest, "bulk", Mock(side_effect=RuntimeError("部分写入失败")))
    with pytest.raises(RuntimeError, match="部分写入失败"):
        ingest.upsert_documents(Mock(), ingest.export_data())


@pytest.mark.parametrize("query,k", [("", 1), ("   ", 1), ("q", 0), ("q", True)])
def test_invalid_search_input_does_not_connect(monkeypatch, query, k):
    connect = Mock(side_effect=AssertionError("非法参数不能连接ES"))
    monkeypatch.setattr(search, "get_es_client", connect)
    with pytest.raises(ValueError):
        search.search_bm25(query, k)
    connect.assert_not_called()


@pytest.mark.parametrize("timed_out,failed", [(True, 0), (False, 1)])
def test_incomplete_bm25_response_raises(monkeypatch, timed_out, failed):
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.search.return_value = {"timed_out": timed_out, "_shards": {"failed": failed}}
    monkeypatch.setattr(search, "get_es_client", lambda: client)
    with pytest.raises(RuntimeError):
        search.search_bm25("HashSet")


def test_bm25_preserves_es_rank_and_never_creates_index(monkeypatch):
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.search.return_value = {
        "timed_out": False,
        "_shards": {"failed": 0},
        "hits": {
            "hits": [
                {
                    "_score": score,
                    "_source": {
                        "document_id": 1,
                        "document_chunk_id": chunk_id,
                        "retrieval_text": f"chunk-{chunk_id}",
                    },
                }
                for chunk_id, score in [(11, 3.0), (10, 1.0)]
            ]
        },
    }
    monkeypatch.setattr(search, "get_es_client", lambda: client)
    hits = search.search_bm25(" HashSet ", candidate_k=2, save_session=False)
    assert [(hit.document_chunk_id, hit.rank, hit.score) for hit in hits] == [
        (11, 1, 3.0),
        (10, 2, 1.0),
    ]
    assert client.search.call_args.kwargs["index"] == "test-ku"
    assert client.search.call_args.kwargs["sort"] == [
        {"_score": "desc"},
        {"knowledge_unit_id": "asc"},
    ]
    client.indices.create.assert_not_called()
    client.indices.delete.assert_not_called()
