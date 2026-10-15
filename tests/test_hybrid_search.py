"""固定候选的离线算法测试，不调用 ES、Chroma 或模型。"""

from copy import deepcopy

import pytest

from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum
from src.retrieval.hybrid_search import fuse_rrf, normalize_hits
from src.retrieval.search_result import SearchResult


def hit(chunk_id, rank, document_id=1, content="相同标题"):
    return SearchResult(
        content=content,
        document_id=document_id,
        document_chunk_id=chunk_id,
        score=0.9,
        rank=rank,
        source_type=SourceTypeEnum.CHROMA_KNOWLEDGE,
        retrieval_type=RetrievalTypeEnum.VECTOR,
    )


def identities(results):
    return [(r.document_id, r.document_chunk_id) for r in results]


def test_rrf_abcd_example():
    vector = [hit(1, 1), hit(2, 2), hit(3, 3)]
    bm25 = [hit(3, 1), hit(1, 2), hit(4, 3)]
    result = fuse_rrf(vector, bm25, top_k=4)
    assert identities(result) == [(1, 1), (1, 3), (1, 2), (1, 4)]
    assert [r.score for r in result] == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 63 + 1 / 61, 1 / 62, 1 / 63]
    )
    assert [r.rank for r in result] == [1, 2, 3, 4]
    assert all(r.source_type is SourceTypeEnum.HYBRID_KNOWLEDGE for r in result)
    assert all(r.retrieval_type is RetrievalTypeEnum.HYBRID for r in result)


def test_duplicate_contributes_once_and_unique_hit_is_reranked():
    vector = [hit(1, 2, content="later"), hit(2, 3), hit(1, 1, content="first")]
    result = fuse_rrf(vector, [hit(1, 1)])
    assert identities(result) == [(1, 1), (1, 2)]
    assert [r.score for r in result] == pytest.approx([2 / 61, 1 / 62])
    assert result[0].content == "first"


@pytest.mark.parametrize("vector_side", [True, False])
def test_single_route_retains_normalized_order(vector_side):
    route = [hit(2, 3), hit(1, 2), hit(1, 1)]
    result = fuse_rrf(route if vector_side else [], [] if vector_side else route)
    assert identities(result) == [(1, 1), (1, 2)]
    assert [r.score for r in result] == pytest.approx([1 / 61, 1 / 62])


def test_both_routes_empty():
    assert fuse_rrf([], []) == []


def test_missing_ids_filtered_and_identical_titles_not_merged():
    route = [
        hit(9, 1, document_id=None),
        hit(None, 2),
        hit(1, 3),
        hit(1, 4, document_id=2),
    ]
    result = fuse_rrf(route, [])
    assert identities(result) == [(1, 1), (2, 1)]
    assert [r.score for r in result] == pytest.approx([1 / 61, 1 / 62])


def test_ties_use_document_then_chunk_ids_and_respect_top_k():
    vector = [hit(3, 1, document_id=2), hit(8, 2)]
    bm25 = [hit(8, 1), hit(3, 2, document_id=2)]
    assert identities(fuse_rrf(vector, bm25, top_k=1)) == [(1, 8)]
    # Same document, same score: chunk ID decides, independent of route order.
    assert identities(fuse_rrf([hit(2, 1)], [hit(1, 1)])) == [(1, 1), (1, 2)]


def test_inputs_unmodified_and_output_objects_copied():
    vector = [hit(2, 3), hit(1, 1)]
    bm25 = [hit(1, 1, content="bm25 text")]
    before = deepcopy((vector, bm25))
    normalized = normalize_hits(vector)
    result = fuse_rrf(vector, bm25)
    assert (vector, bm25) == before
    assert [r.rank for r in normalized] == [1, 2]
    assert all(
        r is not original for r in normalized + result for original in vector + bm25
    )


@pytest.mark.parametrize(
    "name,value", [("top_k", 0), ("top_k", True), ("rrf_k", 0), ("rrf_k", -1)]
)
def test_invalid_fusion_parameters(name, value):
    with pytest.raises(ValueError):
        fuse_rrf([], [], **{name: value})
