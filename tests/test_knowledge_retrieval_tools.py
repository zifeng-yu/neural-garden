"""知识检索工具到原文读取的离线契约测试。"""

import json
import sqlite3
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.get_sqlite_connection as connections
from src.repository.create_table import create_tables
from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum
from src.retrieval.search_result import SearchResult
from src.tools import knowledge_retrieval_tools as tools

DOCUMENT_ID = 17
CHUNK_ID = 41
ORIGINAL_TEXT = "狐狸型专家根据具体情况调整判断，愿意承认不确定性。"
CONTENT_HASH = sha256(ORIGINAL_TEXT.encode()).hexdigest()


@pytest.fixture
def database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "knowledge.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        create_tables(conn)
        conn.execute(
            """
            INSERT INTO documents(id, file_name, file_name_hash, content_hash)
            VALUES (?, ?, ?, ?)
            """,
            (DOCUMENT_ID, "专家.md", "file-hash", CONTENT_HASH),
        )
        conn.execute(
            """
            INSERT INTO document_chunks
                (id, document_id, split_no, content, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (CHUNK_ID, DOCUMENT_ID, 1, ORIGINAL_TEXT, CONTENT_HASH),
        )
    monkeypatch.setattr(connections, "SQLITE_TABLE", str(path))
    return path


def test_search_hit_ids_read_original_chunk(database, monkeypatch):
    backend = Mock(
        return_value=[
            SearchResult(
                content="狐狸型专家的知识单元摘要",
                document_id=DOCUMENT_ID,
                document_chunk_id=CHUNK_ID,
                score=0.9,
                rank=1,
                source_type=SourceTypeEnum.CHROMA_KNOWLEDGE,
                retrieval_type=RetrievalTypeEnum.VECTOR,
            )
        ]
    )
    monkeypatch.setattr(tools, "search", backend)

    result = json.loads(tools.search_knowledge.invoke({"query": "  狐狸型专家  "}))
    backend.assert_called_once_with(
        query="狐狸型专家",
        top_k=3,
        show_score=False,
        save_session=False,
        retrieval_mode="vector",
        use_graph=False,
    )
    assert result == {
        "retrieval_mode": "vector",
        "hits": [
            {
                "document_id": DOCUMENT_ID,
                "chunk_id": CHUNK_ID,
                "rank": 1,
                "summary": "狐狸型专家的知识单元摘要",
            }
        ],
    }
    hit = result["hits"][0]
    original = json.loads(
        tools.read_chunk.invoke(
            {"document_id": hit["document_id"], "chunk_id": hit["chunk_id"]}
        )
    )
    assert original == {
        "status": "found",
        "source_id": f"D{DOCUMENT_ID}:C{CHUNK_ID}",
        "document_id": DOCUMENT_ID,
        "chunk_id": CHUNK_ID,
        "content_hash": CONTENT_HASH,
        "text": ORIGINAL_TEXT,
        "truncated": False,
    }
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM search_sessions").fetchone()[0] == 0


def test_search_empty_hits(monkeypatch):
    monkeypatch.setattr(tools, "search", Mock(return_value=[]))
    assert json.loads(tools.search_knowledge.invoke({"query": "没有相关资料"})) == {
        "retrieval_mode": "vector",
        "hits": [],
    }


@pytest.mark.parametrize(
    "document_id,chunk_id",
    [(DOCUMENT_ID + 1, CHUNK_ID), (DOCUMENT_ID, CHUNK_ID + 1)],
)
def test_read_missing_or_wrong_document_returns_not_found(
    database, document_id, chunk_id
):
    assert json.loads(
        tools.read_chunk.invoke({"document_id": document_id, "chunk_id": chunk_id})
    ) == {"status": "not_found"}


def test_long_original_is_truncated_with_full_content_hash(database):
    content = "原文" * 2100
    content_hash = sha256(content.encode()).hexdigest()
    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE document_chunks SET content=?, content_hash=? WHERE id=?",
            (content, content_hash, CHUNK_ID),
        )
    result = json.loads(
        tools.read_chunk.invoke({"document_id": DOCUMENT_ID, "chunk_id": CHUNK_ID})
    )
    assert result["status"] == "found"
    assert result["text"] == content[:4000]
    assert result["truncated"] is True
    assert result["content_hash"] == content_hash
