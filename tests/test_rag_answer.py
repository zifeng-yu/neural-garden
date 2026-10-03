from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.rag_answer as rag


@pytest.fixture
def dependencies(monkeypatch):
    # Mock 是可记录调用次数的假函数；替换 rag 模块实际使用的三个名字。
    search = Mock(return_value=[])
    chunks = Mock(return_value=[])
    generate = Mock(return_value=None)
    monkeypatch.setattr(rag, "search", search)
    monkeypatch.setattr(rag, "query_document_chuns_by_ids", chunks)
    monkeypatch.setattr(rag, "generate", generate)
    return search, chunks, generate


def test_no_candidates_does_not_generate(dependencies):
    search, chunks, generate = dependencies
    search.return_value = []
    result = rag.answer_question("知识库里没有候选的问题")

    assert result == {"status": "insufficient_evidence", "claims": [], "sources": []}
    assert generate.assert_not_called() is None


def test_unknown_citation_is_rejected(dependencies):
    search, chunks, generate = dependencies
    search.return_value = [
        SimpleNamespace(
            document_chunk_id=101,
            document_id=7,
            rank=1,
            content="退款规则摘要",
        )
    ]
    chunks.return_value = [
        SimpleNamespace(
            id=101,
            document_id=7,
            content="退款申请目前处于处理中。",
            content_hash="fixture-hash-101",
        )
    ]
    # 只有一个有效原文，会由你的程序编号为 S1；假模型却引用 S99。
    generate.return_value = rag.LLMAnswer(
        status="answered",
        claims=[rag.Claims(text="退款申请目前处于处理中。", source_ids=["S99"])],
    )
    with pytest.raises(RuntimeError):
        rag.answer_question("退款处理到哪一步了？")
