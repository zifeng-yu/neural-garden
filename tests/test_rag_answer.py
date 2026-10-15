from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.rag_answer as rag
from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum
from src.retrieval.search_result import SearchResult


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
        SearchResult(
            document_chunk_id=101,
            document_id=7,
            rank=1,
            content="退款规则摘要",
            score=0.9,
            source_type=SourceTypeEnum.CHROMA_KNOWLEDGE,
            retrieval_type=RetrievalTypeEnum.VECTOR,
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


def test_hybrid_fusion_order_reaches_original_sources_and_final_citations(monkeypatch):
    """真实融合固定两路候选，验证问答下游未继续使用旧向量顺序。"""
    import json

    import src.retrieval.hybrid_search as hybrid

    def candidate(chunk_id, rank, route):
        source_type, retrieval_type = {
            "vector": (SourceTypeEnum.CHROMA_KNOWLEDGE, RetrievalTypeEnum.VECTOR),
            "bm25": (SourceTypeEnum.ES_KNOWLEDGE, RetrievalTypeEnum.KEYWORD),
        }[route]
        return SearchResult(
            content=f"知识单元摘要 {chunk_id}",
            document_id=7,
            document_chunk_id=chunk_id,
            score=0.9,
            rank=rank,
            source_type=source_type,
            retrieval_type=retrieval_type,
        )

    # 向量 A/B/C，BM25 C/A/D；真实 RRF 得到 A/C/B。
    vector = [
        candidate(101, 1, "vector"),
        candidate(102, 2, "vector"),
        candidate(103, 3, "vector"),
    ]
    bm25 = [
        candidate(103, 1, "bm25"),
        candidate(101, 2, "bm25"),
        candidate(104, 3, "bm25"),
    ]
    vector_search = Mock(return_value=vector)
    bm25_search = Mock(return_value=bm25)
    monkeypatch.setattr(hybrid, "search_vector", vector_search)
    monkeypatch.setattr(hybrid, "search_bm25", bm25_search)
    monkeypatch.setattr(
        hybrid, "get_existing_collection", lambda _: SimpleNamespace(count=lambda: 10)
    )
    # 故意按另一顺序返回原文，要求 sources 服从融合排名而非数据库顺序。
    chunks = [
        SimpleNamespace(
            id=i, document_id=7, content=f"固定原文 {i}", content_hash=f"hash-{i}"
        )
        for i in (102, 101, 103)
    ]
    chunk_query = Mock(return_value=chunks)
    monkeypatch.setattr(rag, "query_document_chuns_by_ids", chunk_query)
    model = Mock(
        return_value=rag.LLMAnswer(
            status="answered",
            claims=[rag.Claims(text="固定回答", source_ids=["S1", "S2"])],
        )
    )
    monkeypatch.setattr(rag, "generate", model)

    result = rag.answer_question("固定问题", retrieval_mode="hybrid", top_k=3)

    chunk_query.assert_called_once_with([101, 103, 102])
    vector_search.assert_called_once_with(
        query="固定问题", top_k=9, show_score=False, save_session=False
    )
    bm25_search.assert_called_once_with(
        query="固定问题", candidate_k=9, save_session=False
    )
    expected_sources = [
        {
            "source_id": f"S{rank}",
            "document_id": 7,
            "chunk_id": chunk_id,
            "content_hash": f"hash-{chunk_id}",
            "text": f"固定原文 {chunk_id}",
        }
        for rank, chunk_id in enumerate((101, 103, 102), start=1)
    ]
    prompt = model.call_args.kwargs["user_prompt_content"]
    model_sources = json.loads(prompt.split("资料：", 1)[1].strip())
    assert model_sources == expected_sources
    assert result == {
        "status": "answered",
        "claims": [{"text": "固定回答", "source_ids": ["S1", "S2"]}],
        "sources": expected_sources,
    }
    model.assert_called_once()


@pytest.mark.parametrize("mode,use_graph", [("graph", False), ("hybrid", True)])
def test_rag_accepts_graph_mode_and_forwards_enhancement(monkeypatch, mode, use_graph):
    backend = Mock(return_value=[])
    monkeypatch.setattr(rag, "search", backend)
    result = rag.answer_question(
        " 固定问题 ", top_k=2, retrieval_mode=mode, use_graph=use_graph
    )
    backend.assert_called_once_with(
        query="固定问题",
        top_k=2,
        show_score=False,
        save_session=False,
        retrieval_mode=mode,
        use_graph=use_graph,
    )
    assert result == {"status": "insufficient_evidence", "claims": [], "sources": []}
